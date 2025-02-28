import os
import sys
from src.logger import logging
from src.exceptions import CustomException
import torch
from dotenv import load_dotenv

class Normalize(torch.nn.Module):
    def __init__(self,embed_dim):
        try:
            super().__init__()
            self.scale=torch.nn.Parameter(torch.ones(embed_dim))
            self.shift=torch.nn.Parameter(torch.zeros(embed_dim))
            self.eps=1e-5
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)


    def forward(self,x):
        try:
            mean=x.mean(dim=-1,keepdim=True)
            var=x.var(dim=-1,keepdim=True,unbiased=False)
            x_norm=x-mean/torch.sqrt(var*self.eps)
            x_norm=(x_norm*self.scale) +self.shift
            return x_norm
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)

class GELU_Activation(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        try:
            return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)

class FeedForward(torch.nn.Module):
    def __init__(self):
        try:
            super().__init__()
            load_dotenv()
            emb_dim=int(os.getenv('emb_dim'))
            self.layer=torch.nn.Sequential(
                torch.nn.Linear(emb_dim,emb_dim*4),
                GELU_Activation(),
                torch.nn.Linear(emb_dim*4,emb_dim)
            )
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
    def forward(self,x):
        return self.layer(x)

