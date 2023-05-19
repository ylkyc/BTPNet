import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Gaulikely(nn.Module):
    def __init__(self,z):
        super(Gaulikely, self).__init__()
        self.linear1=nn.Linear(z.size()[1],1,bias=True)
        self.linear2=nn.Linear(z.size()[1],1,bias=True)

    def forward(self,z):
        mu=self.linear1(z)
        sigma=F.softplus(self.linear2(z))
        return mu,sigma

def MonteCarlo (mu,sigma,num_sampling):
    z=torch.tensor([])
    for i in range(mu.size()[0]):
        temp=torch.tensor(np.random.normal(mu[i].data, sigma[i].data, num_sampling)).reshape(-1,1)

        temp=torch.mean(temp).reshape(1,1).float()
        z=torch.cat((z,temp),0)
    return z
