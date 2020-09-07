import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense_L(nn.Module):

    def __init__(self, in_size, out_size,bias=True):
        
        
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.dense(x)
        return out

class TDNN(nn.Module):

    def __init__(self, in_size, out_size, context=1, dilation=1, bias=True):
        super().__init__()
        self.dense    = nn.Sequential(
            nn.Linear(in_size*(context*2+1), out_size, bias=True),
            nn.ReLU(),
        )
        self.context  = context
        self.dilation = dilation

    def forward(self, x):
        _,_,d = x.shape
        pad_size = (0,0,self.context*self.dilation,self.context*self.dilation)
        x = x.unsqueeze(1)
        x = F.pad(x, pad=pad_size, mode='replicate')

        x = F.unfold(
                        x, 
                        (self.context*2+1, d), 
                        stride=(1,d), 
                        dilation=(self.dilation,1)
                    )
        x = x.transpose(1,2).squeeze(1)
        out = self.dense(x)
        return out

class DDAE_01(nn.Module):

    def __init__(self):
        super().__init__()
        self.ddae_encoder = nn.Sequential(
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            Dense_L(257,257*3,bias=True),
        )
        self.linear = nn.Linear(257*3, 257, bias=False)
        self.ddae_decoder = nn.Sequential(
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),          
        )

    def forward(self,spec,emma):

        x = self.ddae_encoder(spec)
        x = self.linear(x)
        out = self.ddae_decoder(x)
        return out
    
class DDAE_02(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.ddae_encoder = nn.Sequential(
            TDNN(275,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            Dense_L(257,257*3,bias=True),
        )
        self.linear = nn.Linear(257*3, 257, bias=False)
        self.ddae_decoder = nn.Sequential(
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),       
        )

    def forward(self,spec,emma):
        x = torch.cat((spec,emma),2)
        x = self.ddae_encoder(x)
        x = self.linear(x)
        out = self.ddae_decoder(x)
        return out
    
class DDAE_03(nn.Module):

    def __init__(self):
        super().__init__()
        self.emma_encoder = nn.Sequential(
            TDNN(18,18,bias=True),
            TDNN(18,18,bias=True),
        )
        self.ddae_encoder = nn.Sequential(
            TDNN(275,257,bias=True),
            TDNN(257,257,bias=True),
#             TDNN(257,257,bias=True),
            Dense_L(257,257*3,bias=True),
        )
        self.linear = nn.Linear(257*3, 257, bias=False)
        self.ddae_decoder = nn.Sequential(
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),       
        )

    def forward(self,spec,emma):

        emma = self.emma_encoder(emma)
        x = torch.cat((spec,emma),2)
        x = self.ddae_encoder(x)
        x = self.linear(x)
        out = self.ddae_decoder(x)
        return out

class DDAE_04(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.spec_encoder = nn.Sequential(
            TDNN(257,257,bias=True),
        )
        self.emma_encoder = nn.Sequential(
            TDNN(18,18,bias=True),
            TDNN(18,18,bias=True),
        )
        self.ddae_encoder = nn.Sequential(
            TDNN(275,257,bias=True),
            TDNN(257,257,bias=True),
            Dense_L(257,257*3,bias=True),
        )
        self.linear = nn.Linear(257*3, 257, bias=False)
        self.ddae_decoder = nn.Sequential(
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),
            TDNN(257,257,bias=True),           
        )

    def forward(self,spec,emma):
        spec = self.spec_encoder(spec)
        emma = self.emma_encoder(emma)
        x = torch.cat((spec,emma),2)
        x = self.ddae_encoder(x)
        x = self.linear(x)
        out = self.ddae_decoder(x)
        return out