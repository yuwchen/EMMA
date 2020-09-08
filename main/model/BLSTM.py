import torch
import torch.nn as nn
import pdb
# import sru

class Conv(nn.Module):

    def __init__(self, in_chan, out_chan, kernal ,kernal_m=3, stride=1, dilation=1, padding=0, groups=1,dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernal, stride=stride, dilation=dilation, padding=padding),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Blstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):

        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):]  #merge_mode = 'sum'
        return out
    
class BLSTM_01(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=257, hidden_size=500, num_layers=3),
            nn.Linear(500, 257, bias=True),
            nn.ReLU(),
#             TimeDistributed(nn.Linear(500, 257, bias=True))
        )
    
    def forward(self,spec,emma):
        x = self.lstm_enc(spec)
        
        return x
    
class BLSTM_02(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=275, hidden_size=500, num_layers=3),
            nn.Linear(500, 257, bias=True),
            nn.ReLU(),
#             TimeDistributed(nn.Linear(500, 257, bias=True))
        )
    
    def forward(self,spec,emma):
        x = torch.cat((spec,emma),2)
        x = self.lstm_enc(x)
        
        return x
    
class BLSTM_03(nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.emma_enc = nn.Sequential(
            Blstm(input_size=18, hidden_size=36, num_layers=3),
            nn.Linear(36, 36, bias=True),
            nn.Linear(36, 36, bias=True),
            nn.ReLU(),
        )

        self.lstm_enc = nn.Sequential(
            Blstm(input_size=293, hidden_size=514, num_layers=2),
            Blstm(input_size=514, hidden_size=257, num_layers=1),
            nn.Linear(257, 257, bias=True),
            nn.ReLU(),
        )
    
    def forward(self,spec,emma):

        emma = self.emma_enc(emma)
        x = torch.cat((spec,emma),2)
        x = self.lstm_enc(x)
        
        return x

class BLSTM_04(nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.emma_enc = nn.Sequential(
            Blstm(input_size=18, hidden_size=18, num_layers=4),
            nn.Linear(18, 18, bias=True),
            nn.ReLU(),
        )
        self.spec_enc = nn.Sequential(
            Blstm(input_size=257, hidden_size=257, num_layers=1),
            nn.Linear(257, 257, bias=True),
            nn.ReLU(),
        )
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=275, hidden_size=514, num_layers=2),
            Blstm(input_size=514, hidden_size=257, num_layers=1),
            nn.Linear(257, 257, bias=True),
            nn.ReLU(),
        )
    
    def forward(self,spec,emma):
        emma = self.emma_enc(emma)
        spec = self.spec_enc(spec)
        x = torch.cat((spec,emma),2)
        x = self.lstm_enc(x)
        
        return x

    
class BLSTM_05(nn.Module):
    
    def __init__(self,):
        super().__init__()
        chan = 16
        ker  = 3
        self.emma_conv = nn.Sequential(
            Conv(1,chan,ker,stride=1,dilation=1,padding=1),
            Conv(chan,chan,ker,stride=1,dilation=1,padding=1),
            Conv(chan,1,ker,stride=1,dilation=1,padding=1)
        )
        self.emma_enc = nn.Sequential(
            Blstm(input_size=90, hidden_size=256, num_layers=2),
            nn.Linear(256, 256, bias=True),
            nn.ReLU()
        )

        self.lstm_enc = nn.Sequential(
            Blstm(input_size=256, hidden_size=256, num_layers=3),
            nn.Linear(256, 513, bias=True),
            nn.ReLU(),
        )

    
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.emma_conv(x).squeeze(1)
        encoder = self.emma_enc(x)
        out = self.lstm_enc(encoder)
        return out
    
class BLSTM_06(nn.Module):
    
    def __init__(self,):
        super().__init__()
        chan = 16
        ker  = 3
        self.emma_conv = nn.Sequential(
            Conv(1,chan,ker,stride=1,dilation=1,padding=1),
            Conv(chan,chan,ker,stride=1,dilation=1,padding=1),
            Conv(chan,1,ker,stride=1,dilation=1,padding=1)
        )
        self.emma_enc = nn.Sequential(
            Blstm(input_size=90, hidden_size=256, num_layers=1),
            nn.Linear(256, 256, bias=True),
            nn.ReLU()
        )
        self.spec_conv = nn.Sequential(
            Conv(1,chan,ker,stride=(1,2),dilation=1,padding=1),
            Conv(chan,chan,ker,stride=(1,2),dilation=1,padding=1),
            Conv(chan,chan,ker,stride=(1,2),dilation=1,padding=1),
            Conv(chan,chan,ker,stride=(1,2),dilation=1,padding=1),
            Conv(chan,chan,ker,stride=(1,2),dilation=1,padding=1),
            Conv(chan,chan,ker,stride=(1,2),dilation=1,padding=1)
        )
        self.spec_enc = nn.Sequential(
            Blstm(input_size=144, hidden_size=256, num_layers=2),
            nn.Linear(256, 256, bias=True),
            nn.ReLU()
        )
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=256, hidden_size=256, num_layers=3),
            nn.Linear(256, 513, bias=True),
            nn.ReLU(),
        )

    
    def forward(self,x):

        if x.shape[-1]==90:
            x = x.unsqueeze(1)
            x = self.emma_conv(x).squeeze(1)
            encoder = self.emma_enc(x)
        elif x.shape[-1]==513:
            x = x.unsqueeze(1)
            x = self.spec_conv(x).permute(0,2,1,3)
            x = x.reshape(x.shape[0],x.shape[1],-1)
            encoder = self.spec_enc(x)

        out = self.lstm_enc(encoder)
        return encoder,out
