import argparse,librosa,os,torch,scipy,numpy as np,pdb
from tqdm import tqdm
from util import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_path', type=str, default='')
    parser.add_argument('--clean_path', type=str)
    parser.add_argument('--task', type=str, default='denoise')  # denoise / synthesis 
    parser.add_argument('--out_path', type=str, default='./EMMA_data_pt/')
    parser.add_argument('--pwg_path', type=str, default='./pretrain_model')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    train_path = args.noisy_path
    clean_path = args.clean_path
    config_path = f'{args.pwg_path}/config.yml'
    hdf5_path = f'{args.pwg_path}/stats.h5'
    out_path = args.out_path
    
    #load pwg parameter
    if args.task=='synthesis':
        config = get_config(config_path)
        scaler = get_stats(hdf5_path)
        mel_basis = librosa.filters.mel(config["sampling_rate"], config["fft_size"], config["num_mels"], config["fmin"], config["fmax"])
        torch.save(torch.from_numpy(mel_basis.T),f'{args.pwg_path}/mel_basis.pt')
        torch.save(torch.from_numpy(scaler.mean_),f'{args.pwg_path}/mean.pt')
        torch.save(torch.from_numpy(scaler.scale_),f'{args.pwg_path}/scale.pt')
        mel_basis, mean, scale = torch.load(f'{args.pwg_path}/mel_basis.pt'), torch.load(f'{args.pwg_path}/mean.pt'), torch.load(f'{args.pwg_path}/scale.pt')

    n_frame = 64
    step = 2 if args.task=='denoise' else 1
    
    #generate noisy file
    if args.task=='denoise':
        noisy_files = get_filepaths(train_path)
        for wav_file in tqdm(noisy_files):
            wav,sr = librosa.load(wav_file,sr=16000)
            wav_name = wav_file.split('/')[-1]
            noise = wav_file.split(os.sep)[-2]
            snr = wav_file.split(os.sep)[-3]
            nout_path = os.path.join(out_path,'Noisy',snr,noise,wav_name.split(".")[0])
            
            spec = torch.from_numpy(make_spectrum(y=wav)[0]).t()
            for i in np.arange(spec.shape[0]//n_frame):
                nout_name = nout_path+'_'+str(i)+'.pt'
                check_folder(nout_name)
                torch.save( spec[i*n_frame:(i+1)*n_frame] ,nout_name)
    
    #generate clean file
    clean_files = get_filepaths(clean_path)
    for wav_file in tqdm(clean_files):
        wav_name = wav_file.split('/')[-1]
        c_file = os.path.join(clean_path,wav_name)
        c_wav,sr = librosa.load(c_file,sr=16000)
        c_wav = c_wav.astype('float32')
        cout_path = os.path.join(out_path,'clean') 

        cdata = torch.from_numpy(make_spectrum(y=c_wav)[0]).t() if args.task=='denoise' else get_mel(c_wav,config,mel_basis,mean,scale)[0]
        mat = read_emma(c_file.replace('.wav','.mat'),step,args.task)
        
        cdata = pad_data(cdata,mat)
        for i in np.arange(cdata.shape[0]//n_frame):
            cout_name = os.path.join(cout_path,wav_name.split(".")[0]+'_'+str(i)+'.pt')
            check_folder(cout_name)
            torch.save( cdata[i*n_frame:(i+1)*n_frame] ,cout_name)
