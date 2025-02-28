from torch.utils.data import DataLoader
import tiktoken
import sys
from src.logger import logging
from src.exceptions import CustomException
from src.data_prep.tokenization import CustomDataset
from src.utils import get_text

class CustomDataLoad():
    def __init__(self,data,context_len,batch_size,stride,shuffle=True,drop_last=True,workers=0):
        try:
            logging.info('Data loading in progress')
            tokenizer=tiktoken.get_encoding('gpt2')
            dataset=CustomDataset(tokenizer=tokenizer,data=data,context_len=context_len,stride=stride)

            self.data_loader=DataLoader(
                dataset,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=workers,
                batch_size=batch_size)
            logging.info('data loading complete.')

        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
    def get_data_loader(self):
        return self.data_loader

if __name__=='__main__':
    text=get_text('/Users/pranavks/Desktop/Projects/transformer/artifacts')
    print(text[:50])
    dl=CustomDataLoad(text,4,8,4,True,True,0)
    data_load=dl.get_data_loader()
    cur=iter(data_load)
    print(next(cur))
