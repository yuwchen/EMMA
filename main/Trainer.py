import torch.nn as nn
import torch.nn.functional as F
import torch, mkl
import os, sys, time, numpy as np, librosa, scipy, pandas as pd,pdb
from tqdm import tqdm
from util import *


# def progress_bar(epoch, epochs, step, n_step, time, loss, mode):
#     line = []
#     line = f'\rEpoch {epoch}/ {epochs}'
#     loss = loss/step
#     if step==n_step:
#         progress = '='*30
#     else :
#         n = int(30*step/n_step)
#         progress = '='*n + '>' + '.'*(29-n)
#     eta = time*(n_step-step)/step
#     line += f'[{progress}] - {step}/{n_step} |Time :{int(time)}s |ETA :{int(eta)}s  '
#     if step==n_step:
#         line += '\n'
#     sys.stdout.write(line)
#     sys.stdout.flush()

    
class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, writer, model_path, score_path, args):
#         self.step = 0
        self.epoch = epoch
        self.epoch_count = 0
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer


        self.device = device
        self.loader = loader
        self.criterion = criterion

        self.train_loss = 0
        self.val_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.task = args.task
        if args.mode=='train':
            self.train_step = len(loader['train'])
            self.val_step = len(loader['val'])
        self.args = args
            
        config_path = f'{args.pwg_path}/config.yml'
        model_path  = f'{args.pwg_path}/checkpoint-120000steps.pkl'

        self.config = get_config(config_path)
        self.mel_basis = torch.nn.Parameter(torch.load(f'{args.pwg_path}/mel_basis.pt')).to(device)
        self.mean = torch.nn.Parameter(torch.load(f'{args.pwg_path}/mean.pt')).to(device)
        self.scale = torch.nn.Parameter(torch.load(f'{args.pwg_path}/scale.pt')).to(device)
        self.g_model  = get_model(model_path,self.config).to(device)
        for param in self.g_model.parameters():
            param.requires_grad = False
        self.get_output= get_output(self.config,self.g_model,device)

    def save_checkpoint(self,):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)



    def _train_epoch(self):
        self.train_loss = 0
        self.model.train()
        t_start =  time.time()
        step = 0
        if self.args.encode_loss:
            self._train_step = getattr(self,f'_train_step_mode_{self.task}_encode')
        else:
            self._train_step = getattr(self,f'_train_step_mode_{self.task}')
        for data in self.loader['train']:
            step += 1
            self._train_step(data)
            progress_bar(self.epoch,self.epochs,step,self.train_step,time.time()-t_start,loss=self.train_loss,mode='train')
        self.train_loss /= len(self.loader['train'])
        print(f'train_loss:{self.train_loss}')
        
    
#     @torch.no_grad()
    
        

    def _val_epoch(self):
        self.val_loss = 0
        self.model.eval()
        t_start =  time.time()
        step = 0
        if self.args.encode_loss:
            self._val_step = getattr(self,f'_val_step_mode_{self.task}_encode')
        else:
            self._val_step = getattr(self,f'_val_step_mode_{self.task}')
        for data in self.loader['val']:
            step += 1
            self._val_step(data)
            progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='val')
        self.val_loss /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}')
        
        if self.best_loss > self.val_loss:
            self.epoch_count = 0
            print(f"Save model to '{self.model_path}'")
            self.save_checkpoint()
            self.best_loss = self.val_loss

        

    def train(self):
        model_name = self.model.__class__.__name__ 
        while self.epoch < self.epochs and self.epoch_count<15:
            self._train_epoch()
            self._val_epoch()
            self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            self.epoch_count += 1
        self.writer.close()
    
    
    def write_score(self,test_file,test_path,write_wav=False):
        
        self.model.eval()
        step = 2 if self.args.task=='denoise' else 1
        outname  = test_file.replace(f'{test_path}','').replace('/','_')
        if self.args.task=='denoise':
            noisy,sr = librosa.load(test_file,sr=16000)
            wavname = test_file.split('/')[-1].split('.')[0]
            c_file = os.path.join(self.args.test_clean,wavname.split('_')[0],test_file.split('/')[-1])
            clean,sr = librosa.load(c_file,sr=16000)
            n_data,n_phase,n_len = make_spectrum(y = noisy)
            n_data = torch.from_numpy(n_data).t() 
            mat = read_emma(c_file.replace('.wav','.mat'),step,self.args.task)
            n_data = pad_data(n_data,mat).to(self.device).unsqueeze(0).type(torch.float32)
            spec   = n_data[:,:,:257]
            emma   = n_data[:,:,257:]
            pred = self.model(spec,emma).cpu().detach().numpy()
            enhanced = recons_spec_phase(pred.squeeze().transpose(),n_phase,n_len)
            
        elif self.args.task=='synthesis':
            clean,sr = librosa.load(test_file,sr=16000)
            cdata    = get_mel(clean,self.config,self.mel_basis.cpu(),self.mean.cpu(),self.scale.cpu())[0]
            mat      = read_emma(test_file.replace('.wav','.mat'),step,self.args.task)
            cdata    = pad_data(cdata,mat).to(self.device).unsqueeze(0).type(torch.float32)
            emma     = cdata[:,:,513:]
            pred     = self.model(emma)[1] if self.args.encode_loss else self.model(emma)
            spc = torch.expm1(pred)
            mel = torch.matmul(spc, self.mel_basis)
            mel = torch.log10(torch.max(mel,mel.new_ones(mel.size())*1e-10))
            mel = mel.sub(self.mean[None,:]).div(self.scale[None,:])
            enhanced = self.get_output(mel).squeeze().cpu().detach().numpy()
            
        s_pesq, s_stoi = cal_score(clean,enhanced[:len(clean)])
        with open(self.score_path, 'a') as f:
            f.write(f'{outname},{s_pesq},{s_stoi}\n')
            
        if write_wav:
            method = self.model.__class__.__name__
            wav_path = test_file.replace(f'{test_path}',f'./Enhanced/{method}') 
            check_folder(wav_path)
            enhanced = enhanced/abs(enhanced).max()
            librosa.output.write_wav(wav_path,enhanced,sr)

            
    
            
    def test(self):
        # load model
        mkl.set_num_threads(1)
        self.model.eval()
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])
        test_path = self.args.test_noisy if self.args.task=='denoise' else self.args.test_clean
        test_folders = get_filepaths(test_path)
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f:
            f.write('Filename,PESQ,STOI\n')
        for test_file in tqdm(test_folders):
            self.write_score(test_file,test_path,write_wav=True)
        
        data = pd.read_csv(self.score_path)
        pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
        stoi_mean = data['STOI'].to_numpy().astype('float').mean()

        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(pesq_mean),str(stoi_mean)))+'\n')
        
    def _train_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)

        emma = clean[:,:,257:]
        spec = clean[:,:,:257]
        pred = self.model(noisy,emma)
        loss = self.criterion(pred, spec)
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _val_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)

        emma = clean[:,:,257:]
        spec = clean[:,:,:257]
        pred = self.model(noisy,emma)
        loss = self.criterion(pred, spec)
        self.val_loss += loss.item()

        
    def _train_step_mode_synthesis(self, data):
        device = self.device
        emma, spec = data[:,:,513:],data[:,:,:513]
        emma, spec = emma.to(device).type(torch.float32), spec.to(device).type(torch.float32)
        pred        = self.model(emma)
        loss        = self.criterion(pred, spec)
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _val_step_mode_synthesis(self, data):
        device = self.device
        emma, spec = data[:,:,513:],data[:,:,:513]
        emma, spec = emma.to(device).type(torch.float32), spec.to(device).type(torch.float32)
        pred        = self.model(emma)
        loss        = self.criterion(pred, spec)
        self.val_loss += loss.item()
        
    def _train_step_mode_synthesis_encode(self, data):
        device = self.device
        emma, spec = data[:,:,513:],data[:,:,:513]
        emma, spec = emma.to(device).type(torch.float32), spec.to(device).type(torch.float32)
        # stage 1 
        _,pred = self.model(spec)
        loss = self.criterion(pred, spec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # stage 2
        enc_emma, pred = self.model(emma)
        enc_spec, _    = self.model(spec)
        loss = self.criterion(pred, spec)+self.criterion(enc_emma, enc_spec)
        self.train_loss += self.criterion(pred, spec).item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        
    def _val_step_mode_synthesis_encode(self, data):
        device = self.device
        emma, spec = data[:,:,513:],data[:,:,:513]
        emma, spec = emma.to(device).type(torch.float32), spec.to(device).type(torch.float32)
        _, pred = self.model(emma)
        loss = self.criterion(pred, spec)
        self.val_loss += loss.item()
  