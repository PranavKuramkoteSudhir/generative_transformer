import sys

from src.logger import logging
from src.exceptions import CustomException

import torch

def save_model(model,optimizer,path):
    try:
        torch.save({
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
        },path
        )
        logging.info('wieghts and optimizer saved succesfully')
    except Exception as e:
        logging.info('__Error occoured__')
        raise CustomException(e,sys)

def load_model(model,optimizer,path):
    try:
        checkpoint=torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('wieghts and optimizer loaded succesfully')
    except Exception as e:
        logging.info('__Error occoured__')
        raise CustomException(e,sys)

