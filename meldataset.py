import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf
MAX_WAV_VALUE = 32768.0
import skimage
import skimage.filters
import librosa
def pad_to(in_tens,tgt_size):
  pad_v = torch.zeros([tgt_size - in_tens.size(0)],dtype=in_tens.dtype)
  return torch.cat((in_tens,pad_v))


def load_wav(full_path):
    data, sampling_rate = sf.read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', return_complex=False, normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):

        
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0])
                          for x in fi.read().split('\n') if len(x) > 0 and not "IPA" in x]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0])
                            for x in fi.read().split('\n') if len(x) > 0 and not "IPA" in x]
    
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, pre_blur, blur_sigma=1.0729, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.pre_blur = pre_blur
        self.blur_sigma = blur_sigma
        self.bad_indexes = []
        self.num_resample_warns = 0
        self.num_resample_warns_max = 10
        if fine_tuning:
            print(f"Load mels from {base_mels_path}")

    def __getitem__(self, index):
        if index in self.bad_indexes:
            return self.__getitem__(index + 1)
            
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            try:
                filen, ext = os.path.splitext(filename)
                file_p_path = filename.replace(ext, "_p.ogg")
                if os.path.isfile(file_p_path):
                    filename = file_p_path
                    
                audio, sampling_rate = load_wav(filename)
                if sampling_rate != self.sampling_rate:
                    if self.num_resample_warns < self.num_resample_warns_max:
                        print(f"File SR {sampling_rate} != {self.sampling_rate}.. resampling and re-saving to {file_p_path}\nthis will only print {self.num_resample_warns_max} times")
                        self.num_resample_warns += 1
                    
                    audio = librosa.resample(audio, sampling_rate, self.sampling_rate, res_type="kaiser_fast")
                    sampling_rate = self.sampling_rate
                    sf.write(file_p_path, audio, sampling_rate)
                    
            except KeyboardInterrupt:
                return None
            except:
                print(f"Could not open file {filename}")
                self.bad_indexes.append(index)
                return self.__getitem__(index + 1)
                
            
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
            if self.pre_blur:
                mel = torch.from_numpy(
                    skimage.filters.gaussian(mel.squeeze().cpu().numpy(), 
                                             sigma=self.blur_sigma, channel_axis=0)).unsqueeze(0)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            
            mel = torch.from_numpy(mel)
            if torch.isnan(mel).any():
                raise ValueError(f"NaN in mel {filename}")

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        
        audio_sq = audio.squeeze(0)
        
        if len(audio_sq) != self.segment_size and self.split:
            if len(audio_sq) < self.segment_size:
                audio_sq = pad_to(audio_sq,self.segment_size)
            else:
                audio_sq = audio_sq[:self.segment_size]
            
            
            audio = audio_sq.unsqueeze(0)
        
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        mel_ret, audio_ret, ml_ret = mel.squeeze(), audio.squeeze(0), mel_loss.squeeze()

        return (mel_ret, audio_ret, filename, ml_ret)

    def __len__(self):
        return len(self.audio_files)
