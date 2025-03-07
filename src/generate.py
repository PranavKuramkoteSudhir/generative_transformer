import os
import sys

import torch

from src.logger import logging
from src.exceptions import CustomException
import tiktoken

def token_to_text(token_ids, tokenizer=tiktoken.get_encoding('gpt2')):
    logging.info('running token_to_text')
    try:
        flat = token_ids.squeeze(0) # remove batch dimension
        return tokenizer.decode(flat.tolist())
    except Exception as e:
        logging.info('__Error Occoured__')
        raise CustomException(e,sys)
def text_to_token_ids(text, tokenizer=tiktoken.get_encoding('gpt2')):
    logging.info('running text_to_token_ids')
    try:
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor
    except Exception as e:
        logging.info('__Error Occoured__')
        raise CustomException(e,sys)

def generate_text(model,idx,max_new_tokens,context_size,temperature=0,top_k=None,eos_id=None):
    logging.info('running generate_text')
    try:
        context=idx[:,-context_size:]
        for i in range(max_new_tokens):
            context=context[:,-context_size:]
            with torch.no_grad():
                logits=model(context)
                logits=logits[:,-1,:]
            if top_k is not None:
                top_logits,_=torch.topk(logits,top_k)
                min_val=top_logits[:,-1]
                logits=torch.where(min_val>logits,torch.tensor(float('-inf')).to(logits.device),logits)
            if temperature>0.0:
                logits=logits/temperature
                prob=torch.softmax(input=logits,dim=-1)
                next_index=torch.multinomial(prob,num_samples=1)
            else:
                next_index=torch.argmax(logits,dim=-1,keepdim=True)
            if eos_id==next_index:
                break
            logging.info(f"Generated Token: {next_index.item()} -> {token_to_text(next_index)}")
            context=torch.cat([context,next_index],dim=1)
            
        return context
    except Exception as e:
        logging.info('__Error Occoured__')
        raise CustomException(e,sys)


