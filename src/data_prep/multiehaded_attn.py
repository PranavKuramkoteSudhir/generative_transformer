from src.logger import logging
from src.exceptions import CustomException
import torch
import sys

class multihead_attn(torch.nn.Module):
    def __init__(self,dIn,dOut,attnHeads,context,dropout=0.1,qkv_bias=False):
        try:
            super().__init__()
            if dOut%attnHeads!=0:
                logging.info('The Dimension out should be divisble by no. of attention heads. Function failed to execute')
            self.dIn=dIn
            self.dOut=dOut
            self.attnHeads=attnHeads
            self.head_dim=self.dOut//self.attnHeads
            self.context=context
            self.dropout=torch.nn.Dropout(dropout)
            self.bias=qkv_bias
            self.query=torch.nn.Linear(self.dIn,self.dOut,self.bias)
            self.key=torch.nn.Linear(self.dIn,self.dOut,self.bias)
            self.value=torch.nn.Linear(self.dIn,self.dOut,self.bias)
            self.project=torch.nn.Linear(self.dOut,self.dOut)
            self.register_buffer('mask',torch.triu(torch.ones(self.context,self.context),diagonal=1))
        except Exception as e:
            logging.info('__Error Occoured__')
            raise CustomException(e,sys)


    def forward(self,x):
        try:
            batch,num_tokens,dIn=x.shape
            q=self.query(x)
            k=self.key(x)
            v=self.value(x)
            q=q.view(batch,num_tokens,self.attnHeads,self.head_dim) #Split the columns to another dimension
            k=k.view(batch,num_tokens,self.attnHeads,self.head_dim)
            v=v.view(batch,num_tokens,self.attnHeads,self.head_dim)
            q=q.transpose(1,2)
            k=k.transpose(1,2)
            v=v.transpose(1,2)

            scores= q @ k.transpose(2,3)

            bool_mask=self.mask.bool()[:num_tokens,:num_tokens]

            scores.masked_fill_(bool_mask,-torch.inf)

            weights=torch.softmax(scores/(k.shape[-1])**0.5,dim=-1)
            weights=self.dropout(weights)

            context_vector=(weights @ v)
            context_vector=context_vector.transpose(1,2)

            context_vector=context_vector.contiguous().view(batch,num_tokens,self.dOut)
            context_vector=self.project(context_vector)
            return context_vector
        except Exception as e:
            logging.info('__Error Occoured__')
            raise CustomException(e,sys)

if __name__=='__main__':
    torch.manual_seed(123)

    # Define the tensor with 3 rows and 6 columns
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
        [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
        [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
    )

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) 

    batch_size, context_length, d_in = batch.shape
    d_out = 6
    mha = multihead_attn(d_in, d_out,2,context_length, 0.0,False )
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)