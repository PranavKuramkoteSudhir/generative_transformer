import os
import sys

from dotenv import load_dotenv
import torch

from src.logger import logging
from src.exceptions import CustomException

from src.GPT.model_forward import GPTModel
from src.data_prep.tokenization import CustomDataset
from src.data_prep.data_load import CustomDataLoad
from src.utils import get_text
from src.generate import generate_text,token_to_text,text_to_token_ids
from src.checkpoint import save_model

def calc_loss(input,target,model,device):
    logging.info('running calc_loss')
    try:
        input,target=input.to(device),target.to(device)
        logits=model(input)
        return torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten())
    except Exception as e:
        logging.info('__Error occoured__')
        raise CustomException(e,sys)

def calc_loss_loader(data_loader,model,device,num_iters=None):
    logging.info('running calc_loss_loader')
    try:
        
        total_loss=0
        if len(data_loader)==0:
            logging.info('__data_loader empty__')
            return float('nan')
        elif num_iters is None:
            logging.info('Number of iterators not mentioned considering length of data loaders as num of iterators')
            num_iters=len(data_loader)
        else:
            num_iters=min(num_iters,len(data_loader))
        for i,(input,target) in enumerate(data_loader):
            if i<num_iters:
                loss=calc_loss(input,target,model,device)
                total_loss+=loss.item()
            else:
                break
        return total_loss/num_iters
    except Exception as e:
        logging.info('__Error occoured__')
        raise CustomException(e,sys)


def model_eval(model,train_loader,val_loader,device,eval_iteration):
    logging.info('running model_eval')
    try:
        model.eval()
        with torch.no_grad():
            trainloss=calc_loss_loader(train_loader,model,device,eval_iteration)
            valloss=calc_loss_loader(val_loader,model,device,eval_iteration)
        return trainloss,valloss
    except Exception as e:
        logging.info('__Error occoured__')
        raise CustomException(e,sys)
    
    

def training_loop(model,train_loader,val_loader,optimizer,device,epochs,evaluation_frequency,evaluation_iteration):
    logging.info('running training_loop')
    try:
        load_dotenv()
        training_loss=[]
        validation_loss=[]
        list_tokens_seen=[]

        num_tokens=0
        global_step=-1

        for epoch in range(epochs):
            model.train()
            for input, target in train_loader:
                optimizer.zero_grad()
                loss = calc_loss(input,target,model,device)
                loss.backward()
                optimizer.step()
                num_tokens+=input.numel()
                global_step+=1
                if global_step%evaluation_frequency==0:
                    train_loss,val_loss=model_eval(model,train_loader,val_loader,device,evaluation_iteration)
                    training_loss.append(train_loss)
                    validation_loss.append(val_loss)
                    list_tokens_seen.append(num_tokens)
                    logging.info(f'\n__Epoch: {epoch+1} Step:{global_step}__\nTraining loss: {train_loss}\nValidation_loss: {val_loss}')
        result=''
        res=[]
        for i in range(len(training_loss)):
            
            s=f'Training loss: {training_loss[i]}    Validation loss: {validation_loss[i]} Number of tokens seen: {list_tokens_seen[i]}'  
            res.append(s)
        result='\n'.join(res)

        logging.info(f'\n________________Training complete________________\nSummery:\n{result}')
        return training_loss,validation_loss,list_tokens_seen
    except Exception as e:
        logging.info('__Error occoured__')
        raise CustomException(e,sys)

if __name__=='__main__':
    data=get_text("/Users/pranavks/Desktop/Projects/transformer/artifacts")
    train=data[int(len(data)*0.8):]
    test=data[:-int(len(data)*0.2)]
    load_dotenv()
    
    dl_train=CustomDataLoad(
        train,
        int(os.getenv('context_len')),
        int(os.getenv('batch_size')),
        int(os.getenv('stride')),
        shuffle=False,
        drop_last=True,
        workers=4)
    dl_test=CustomDataLoad(
        test,
        int(os.getenv('context_len')),
        int(os.getenv('batch_size')),
        int(os.getenv('stride')),
        shuffle=False,
        drop_last=True,
        workers=4)
    data_load_train=dl_train.get_data_loader()
    data_load_test=dl_test.get_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=GPTModel()
    model.to(device)
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.01)
    epochs=5

    training_loop(model,data_load_train,data_load_test,optimizer,device,epochs,5,evaluation_iteration=5)
    save_model(model,optimizer,'artifacts/models/model_and_optimizer.pth')
    starter="This came as a surprize to alice she couldn't believe her eyes she didn't understand what was going one she was shocked to say the least"
    generated_tokens=generate_text(model,text_to_token_ids(text=starter),100,int(os.getenv('context_len')),1.4,500,None)
    generated_text=token_to_text(token_ids=generated_tokens)
    logging.info(generated_text)



