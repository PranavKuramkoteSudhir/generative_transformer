from torch.utils.data import DataLoader
import tiktoken
import sys
from src.logger import logging
from src.exceptions import CustomException
from src.data_prep.tokenization import CustomDataset
from src.utils import get_text
from src.generate import token_to_text
#Everything in working condition
class CustomDataLoad():
    def __init__(self,data,context_len,batch_size,stride,shuffle=True,drop_last=True,workers=0):
        try:
            logging.info('Data loading in progress')
            tokenizer=tiktoken.get_encoding('gpt2')
            #Load dataset in batches and use the provided tokenizer to encode all the tokens
            dataset=CustomDataset(tokenizer=tokenizer,data=data,context_len=context_len,stride=stride)
            #Create Data loader class and override it
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
    dl=CustomDataLoad(text,4,1,2,False,True,0)
    data_load=dl.get_data_loader()
    cur=iter(data_load)
    input,target=next(cur)
    print("Shape before slicing:", input.shape)
    print(token_to_text(input[0:1]))
    print(token_to_text(target[0:1]))
