import os
import sys

from src.logger import logging
from src.exceptions import CustomException

import torch
from dotenv import load_dotenv

from src.transformer.transformer import Transformer
from src.transformer.components.transformer_components import Normalize

class GPTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.token_emb=torch.nn.Embedding(int(os.getenv('vocab_size')),int(os.getenv('emb_dim')))
        self.positional_emb=torch.nn.Embedding(int(os.getenv('context_len')),int(os.getenv('emb_dim')))
        self.drop=torch.nn.Dropout(float(os.getenv('drop_rate')))

        self.transformer_layers=torch.nn.Sequential(*[Transformer() for _ in range(int(os.getenv('n_layers')))])

        self.norm=Normalize(int(os.getenv('emb_dim')))
        self.out=torch.nn.Linear(int(os.getenv('emb_dim')),int(os.getenv("vocab_size")),bias=False)

    def forward(self,input):
        batch,sequence_len=input.shape
        x_token=self.token_emb(input)
        x_pos=self.positional_emb(torch.arange(sequence_len,device=input.device))
        x=x_token+x_pos
        x=self.drop(x)
        x=self.transformer_layers(x)
        x=self.norm(x)
        logits=self.out(x)
        return logits


if __name__=='__main__':
    torch.manual_seed(123)
    input = torch.randint(0, int(os.getenv('vocab_size')), (2, 4))
    model=GPTModel()
    output=model(input)
    print(output.shape)








