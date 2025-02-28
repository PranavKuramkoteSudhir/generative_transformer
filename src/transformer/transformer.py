import os
import sys
from src.logger import logging
from src.exceptions import CustomException
from dotenv import load_dotenv
import torch

from src.data_prep.multiehaded_attn import Multihead_attn
from src.data_prep.transformer_components import Normalize, GELU_Activation, FeedForward 

class Transformer(torch.nn.Module):
    def __init__(self):
        try:
            super().__init__()
            load_dotenv()
            self.attn=Multihead_attn(
                dIn=int(os.getenv('emb_dim')),
                dOut=int(os.getenv('emb_dim')),
                attnHeads=int(os.getenv('n_heads')),
                context=int(os.getenv('context_len')),
                dropout=float(os.getenv('drop_rate')),
                qkv_bias=os.getenv("qkv_bias", "False").lower() == "true")
            self.norm1=Normalize(int(os.getenv('emb_dim')))
            self.norm2=Normalize(int(os.getenv('emb_dim')))
            self.ff=FeedForward()
            self.gelu=GELU_Activation()
            self.dropout=torch.nn.Dropout(float(os.getenv('drop_rate')))
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
    
    def forward(self,x):
        try:
            short=x
            x=self.norm1(x)
            x=self.attn(x)
            x=self.dropout(x)
            x=x+short

            short=x
            x=self.norm2(x)
            x=self.ff(x)
            x=self.dropout(x)
            x=x+short
            return x
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
if __name__=='__main__':
    torch.manual_seed(123)
    x=torch.rand(2,4,768)
    t=Transformer()
    output=t(x)
    print(output.shape)





