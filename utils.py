import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
from istftnetfe import ISTFTNetFE

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def save_ts_gen(folpath, gen, g_stft, sr):
  device_before = next(gen.parameters()).device
  print(f"Exporting TorchScript generator to {folpath}")
  with torch.no_grad():
    g_fe = ISTFTNetFE(gen, g_stft)
    g_fe.export_ts(folpath, sr)
  
  # return the gen and stft to their devices to avoid errors at next iter
  gen = gen.to(device_before)
  g_stft = g_stft.to(device_before)
  g_stft.window = g_stft.window.to(device_before)
  print("Done.")
  return folpath

  
  


