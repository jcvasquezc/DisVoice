import torch
from torch import nn, optim
import torch.nn.functional as F



class RAEenc(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.lstm1=nn.LSTM(128,64, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256, dim)

    def forward(self, x):
        x=x[:,0,:,:]
        x=x.permute(0,2,1)
        x,(hn,cn)=self.lstm1(x)
        #print(hn.size())
        hn=hn.permute(1,0,2)
        #print(hn.size()) 
        x=x[:,-1,:]
        hn=hn.contiguous().view(hn.size(0),-1)
        x = x.view(x.size(0), -1)
        x2=torch.cat((x,hn),1)
        x = F.leaky_relu(self.linear(x2))
        
        
        return x

class RAEdec(nn.Module):
    def __init__(self, dim=32, seq_len=126):
        super().__init__()
        self.lstm = nn.LSTM(dim,128, batch_first=True, num_layers=2)
        self.seq_len=seq_len

    def forward(self, x):
        x = torch.cat([x] *  self.seq_len, 1).view(x.size(0), self.seq_len, x.size(1))
        x, (h,c)=self.lstm(x)
        x=x.permute(0,2,1)
        x=x.view(x.size(0), 1, x.size(1), x.size(2))
        return x


class RAEn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = RAEenc(dim=dim)
        self.decoder = RAEdec(dim=dim)

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return x, bottleneck

