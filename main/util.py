import librosa,os,pdb
import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
import scipy, sklearn
import torch.nn.functional as F
import torch
import torch.nn as nn
import yaml,torch,librosa,numpy as np,sys
sys.path.append("..") 
from parallel_wavegan.utils import read_hdf5
from parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator
from sklearn.preprocessing import StandardScaler



epsilon = np.finfo(float).eps

def read_emma(emma_path,step=2,mode='synthesis'):

    mat = scipy.io.loadmat(emma_path)
    emma = np.asarray(mat['mxy'])
    emma = np.array(emma.T)
    #emma = align_signals(emma)
    emma = sklearn.preprocessing.normalize(emma, norm="max",axis=0)
    emma = torch.from_numpy(emma[0::step,2:20])
    if mode=='synthesis':
        x = emma.unsqueeze(0)
        _,_,d = x.shape
        x = x.unsqueeze(1)
        pad_size = (0,0,2,2)
        x = F.pad(x, pad=pad_size, mode='replicate')
        x = F.unfold(x, (2*2+1, d), stride=(4,d) )
        emma = x.transpose(1,2).squeeze()
    return emma


def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def cal_score(clean,enhanced):
    clean = clean/abs(clean).max()
    enhanced = enhanced/abs(enhanced).max()
#     pdb.set_trace()
    s_stoi = stoi(clean, enhanced, 16000)
    s_pesq = pesq(clean, enhanced, 16000)
    
    return round(s_pesq,5), round(s_stoi,5)


def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def pad_data(spec, mat):
    pad_size = (0,0,0,spec.shape[0]-mat.shape[0])
    mat = F.pad(mat.unsqueeze(0).unsqueeze(0), pad=pad_size,mode='replicate').squeeze()
    return torch.cat((spec,mat),dim=-1)


def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    # y = y / np.max(abs(y)) / 2.

    D = librosa.stft(y,center=True, n_fft=512, hop_length=128,win_length=512,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1

    return Sxx, phase, len(y)

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**Sxx_r)

    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     center=True,
                     hop_length=128,
                     win_length=512,
                     window=scipy.signal.hamming,
                     length=length_wav)
    return result


def get_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config['trim_silence'] = False
    return config

def get_stats(hdf5_path):
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(hdf5_path, "mean")
    scaler.scale_ = read_hdf5(hdf5_path, "scale")
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler

def get_model(model_path,config):
    model = ParallelWaveGANGenerator(**config["generator_params"])
    model.load_state_dict(torch.load(model_path, map_location="cpu")["model"]["generator"])
    model = model.eval().cuda()
    return model

class get_output(nn.Module):
    def __init__(self,config,model,device):
        super().__init__()
        self.model  = model.to(device)
        self.device = device
        self.hop_size = config["hop_size"]
    def forward(self,c):

        z = torch.randn(c.shape[0], 1, c.shape[1] * self.hop_size).to(self.device)
        c = F.pad(c.permute(0,2,1),(2,2),'reflect')
        y = self.model(z,c)
        return y

def specn(audio,mel_basis,fft_size, hop_size, window):
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, window=window, pad_mode="reflect")
    phase = np.exp(1j * np.angle(x_stft))
    spc = np.abs(x_stft).T
    spc1p = torch.log1p(torch.from_numpy(spc))
    spc = torch.expm1(spc1p)
    return spc1p,torch.log10(torch.mm(spc, mel_basis))

def get_mel(data,config,mel_basis,mean,scale):
    spec1p, mel = specn(data,mel_basis,hop_size=config["hop_size"],
                           fft_size=config["fft_size"],
                           window=config["window"],
                           )
    audio = np.pad(data, (0, config["fft_size"]), mode="reflect")
    audio = audio[:len(mel) * config["hop_size"]]
    mel = (mel-mean[None,:])/scale[None,:]
    return spec1p,mel,audio

def progress_bar(epoch, epochs, step, n_step, time, loss, mode):
    line = []
    line = f'\rEpoch {epoch}/ {epochs}'
    loss = loss/step
    if step==n_step:
        progress = '='*30
    else :
        n = int(30*step/n_step)
        progress = '='*n + '>' + '.'*(29-n)
    eta = time*(n_step-step)/step
    line += f'[{progress}] - {step}/{n_step} |Time :{int(time)}s |ETA :{int(eta)}s  '
    if step==n_step:
        line += '\n'
    sys.stdout.write(line)
    sys.stdout.flush()

