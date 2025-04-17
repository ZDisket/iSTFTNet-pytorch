import torch.nn as nn
import torch
from typing import Optional
import torch.nn.functional as F

class TransposeLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TransposeLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps, affine)

    def forward(self, x):
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Apply LayerNorm
        x = self.ln(x)
        # Transpose back from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        return x

class APTx(nn.Module):
    """
    APTx: Alpha Plus Tanh Times, an activation function that behaves like Mish,
    but is 2x faster.

    https://arxiv.org/abs/2209.06119
    """

    def __init__(self, alpha=1, beta=1, gamma=0.5, trainable=False):
        """
        Initialize APTx initialization.
        :param alpha: Alpha
        :param beta: Beta
        :param gamma: Gamma
        :param trainable: Makes beta and gamma trainable, dynamically optimizing the upwards slope and scaling
        """
        super(APTx, self).__init__()
        self.alpha = alpha
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.beta = beta
            self.gamma = gamma

    def forward(self, x):
        return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x

def masked_fill_(x: torch.Tensor, mask: torch.Tensor, fill_value: float):
    """
    In-place masked fill. Where mask == True, fill `x` with `fill_value`.

    x:    (B, C, L)
    mask: (B, 1, L)  (True means invalid/padded)
    """
    # We broadcast the mask across the channel dimension:
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1)  # shape: (B, C, L)
    x.masked_fill_(mask, fill_value)

def masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes max over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    x:    (B, C, L)
    mask: (B, 1, L) (True means invalid/padded)
    """
    # Temporarily fill invalid positions with a very small number so they won't dominate max
    x_clone = x.clone()
    masked_fill_(x_clone, mask, float('-inf'))  # fill padded positions with -inf
    # Max over seq_len dimension
    max_vals, _ = x_clone.max(dim=-1, keepdim=True)  # shape: (B, C, 1)
    return max_vals


def masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes average over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    x:    (B, C, L)
    mask: (B, 1, L) (True means invalid/padded)
    """
    # We will sum only valid positions and then divide by count of valid positions
    x_clone = x.clone()

    # Invert the mask to get a "valid" mask where True = valid
    valid_mask = ~mask  # shape: (B, 1, L)
    if valid_mask.shape[1] == 1 and x_clone.shape[1] != 1:
        valid_mask = valid_mask.expand(-1, x_clone.shape[1], -1)  # (B, C, L)

    # Fill invalid positions with zero, so they don't contribute to sum
    x_clone.masked_fill_(~valid_mask, 0.0)

    # Sum across seq_len
    sum_vals = x_clone.sum(dim=-1, keepdim=True)  # shape: (B, C, 1)

    # Count of valid positions per (B, C)
    counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # shape: (B, C, 1)

    # Weighted average
    avg_vals = sum_vals / counts
    return avg_vals


class SAM1D(nn.Module):
    """
    Spatial Attention Module for 1D sequences.
    - Takes (B, C, L) as input
    - Produces a spatial attention map of shape (B, 1, L)
    - Then multiplies it (elementwise) by the original input (while respecting the mask).
    """

    def __init__(self, bias=False):
        super(SAM1D, self).__init__()
        self.bias = bias
        # We take 2 'channels' as input (max, avg along channel dim),
        # and produce a 1-channel output (spatial attention).
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            bias=self.bias
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, L)
        mask: (B, 1, L) (True means invalid/padded)
        """
        # Masked max & avg across channel dimension
        max_out = x.max(dim=1, keepdim=True)[0]  # shape: (B, 1, L)
        avg_out = x.mean(dim=1, keepdim=True)  # shape: (B, 1, L)

        # Expand the mask to match shape (B, 1, L):
        # We skip that because max_out and avg_out already have shape (B, 1, L).
        # If needed, we can do:
        masked_fill_(max_out, mask, 0.0)  # or a safe fill for the 'max' map
        masked_fill_(avg_out, mask, 0.0)  # fill with 0.0 for 'avg'

        # Concatenate along the channel dimension => shape (B, 2, L)
        concat = torch.cat((max_out, avg_out), dim=1)

        # Convolution over the sequence dimension
        output = self.conv(concat)  # shape: (B, 1, L)

        output = output.masked_fill(mask, -10.0) # prevent masked positions from activating the sigmoid.

        # Sigmoid
        output = torch.sigmoid(output)

        # Multiply elementwise with the original x (but also mask out padded positions)
        # We'll broadcast output from (B, 1, L) to (B, C, L)
        output = output * x

        # Finally, we can mask out padded positions so that they remain zero if desired
        masked_fill_(output, mask, 0.0)

        return output


class CAM1D(nn.Module):
    """
    Channel Attention Module for 1D sequences.
    - Takes (B, C, L) as input
    - Pools across the L dimension (with masked max & avg) to get (B, C, 1)
    - Then uses an MLP (two linear layers) to compute channel attention
    - Finally multiplies it with the input (while respecting the mask).
    """

    def __init__(self, channels, reduction_ratio):
        super(CAM1D, self).__init__()
        self.channels = channels
        self.r = reduction_ratio
        self.linear = nn.Sequential(
            nn.Linear(self.channels, self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.r, self.channels, bias=True)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, L)
        mask: (B, 1, L) (True means invalid/padded)
        """
        # We do masked max pooling & masked avg pooling across the length dimension
        max_pool = masked_max_pool1d(x, mask)  # (B, C, 1)
        avg_pool = masked_avg_pool1d(x, mask)  # (B, C, 1)

        # Flatten for linear => shape (B, C)
        # (We remove the last dimension which is 1)
        max_pool_flat = max_pool.squeeze(-1)  # (B, C)
        avg_pool_flat = avg_pool.squeeze(-1)  # (B, C)

        # Feed through MLP
        mlp_max = self.linear(max_pool_flat)  # (B, C)
        mlp_avg = self.linear(avg_pool_flat)  # (B, C)

        # Sum
        attn = mlp_max + mlp_avg  # (B, C)

        # Channel attention map
        attn = torch.sigmoid(attn).unsqueeze(-1)  # shape (B, C, 1)

        # Multiply by the original input (broadcast across L dimension)
        output = attn * x

        # We can optionally mask out the padded positions
        masked_fill_(output, mask, 0.0)

        return output


class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module for 1D sequences.
    - Applies Channel Attention (CAM1D)
    - Then applies Spatial Attention (SAM1D)
    - Adds the result to the original input as a residual connection.
    """

    def __init__(self, channels, reduction_ratio=8):
        super(CBAM1D, self).__init__()
        self.cam = CAM1D(channels, reduction_ratio)
        self.sam = SAM1D(bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, L)
        mask: (B, 1, L) (True means invalid/padded)
        """
        # Channel Attention
        out = self.cam(x, mask)
        # Spatial Attention
        out = self.sam(out, mask)
        # Residual connection
        return out + x

class CausalConv1d(nn.Conv1d):
    """
    A 1D convolution layer that applies causal padding on the left side.
    For a kernel size k and dilation d, it pads the input with d*(k-1) zeros on the left.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        # Compute the required left-padding for causal convolution
        self.causal_padding = dilation * (kernel_size - 1)
        # Remove any padding passed in kwargs and use padding=0 (we handle it manually)
        kwargs.pop('padding', None)
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, padding=0, **kwargs)

    def forward(self, x):
        # Pad only on the left: (left, right)
        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))
        return super().forward(x)

class ResidualBlock1D(nn.Module):
    """
    Conv1D+Squeeze-Excite+LayerNorm residual block for sequence modeling with optional masking.

    New parameter:
      stride: int
        if >1  → use nn.Conv1d(..., stride=stride)
        if ==1 → use plain nn.Conv1d(..., stride=1)
        if <0  → use nn.ConvTranspose1d(..., stride=abs(stride), output_padding=abs(stride)-1)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        dropout: float = 0.3,
        act: str = "relu",
        causal: bool = False,
    ):
        super().__init__()
        self.stride = stride

        # --- conv1: may be downsample, upsample, or normal ---
        if causal:
            # assume CausalConv1d signature supports stride
            self.conv1 = CausalConv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride
            )
        else:
            if stride > 1:
                # strided down‐sampling
                self.conv1 = nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    padding="same"
                )
            elif stride < 0:
                # transposed conv for up‐sampling
                pad = dilation * (kernel_size - 1) // 2
                self.conv1 = nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=abs(stride),
                    padding=pad,
                    output_padding=abs(stride) - 1
                )
            else:
                # no length change
                self.conv1 = nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=1,
                    padding="same"
                )

        # conv2 always keeps length unchanged
        if causal:
            self.conv2 = CausalConv1d(out_channels, out_channels,
                                      kernel_size=kernel_size,
                                      dilation=dilation)
            self.cbam = None
        else:
            self.conv2 = nn.Conv1d(
                out_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same"
            )
            self.cbam = CBAM1D(out_channels)

        # layer norms, activation, dropout
        self.norm1 = TransposeLayerNorm(out_channels)
        self.norm2 = TransposeLayerNorm(out_channels)
        self.relu = APTx() if act == "aptx" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # --- residual path: match both channel & length changes ---
        if in_channels != out_channels or stride != 1:
            if stride > 1:
                self.residual = nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride
                )
            elif stride < 0:
                self.residual = nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=abs(stride),
                    output_padding=abs(stride) - 1
                )
            else:
                self.residual = nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1
                )
        else:
            self.residual = nn.Identity()

    def forward(self, x, x_mask: Optional[torch.Tensor] = None):
        # residual connection
        res = self.residual(x)

        # first conv + norm + (optional masking) + act + dropout
        out = self.conv1(x)
        out = self.norm1(out)
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.relu(out)
        out = self.dropout(out)

        # second conv + norm + CBAM + add residual
        out = self.conv2(out)
        out = self.norm2(out)
        if self.cbam is not None:
            out = self.cbam(out, x_mask)
        out = out + res

        # final activation + dropout (apply mask again if provided)
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.relu(out)
        out = self.dropout(out)

        return out
