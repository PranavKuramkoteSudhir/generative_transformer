from torch.utils.data import Dataset
import torch
from src.logger import logging
from src.exceptions import CustomException
import sys

"""

Create tokens and batch the data into multiple batches to feed into the model

"""

class CustomDataset(Dataset):   
    def __init__(self,tokenizer,data,context_len,stride):
        try:
            self.input_id=[]
            self.target_id=[]
            logging.info('starting to encode data')
            tokens=tokenizer.encode(data,allowed_special='<EOT>')

            logging.info('Creating input and target chunks')
            for i in range(0,len(tokens)-context_len,stride):
                params_chunk=data[i:i+context_len]
                target_chunk=data[i+1:i+context_len+1]
                self.input_id.append(torch.tensor(params_chunk))
                self.target_id.append(torch.tensor(target_chunk))
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
            
    def __len__(self):
        try:
            return (self.input_id)
        
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
        
    def __getitem__(self, index):
        try:
            return self.input_id[index],self.target_id[index]
        
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
