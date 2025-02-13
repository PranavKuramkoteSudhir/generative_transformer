import re
import os
import sys
from src.logger import logging
from src.exceptions import CustomException
from src.utils import get_text

"""

This is just a demonstration for the understanding of BPE tokenization, I will end up using the tiktokenizer while working on the model anyways

"""

class Tokenizer:
    """
    
    takes text input and converts it to tokens and asigns numbers to each unique word returns encoder and decoder in a tuple.
    decoder just contains encoders key as vaue and value as key

    """
    def transcode(self,txt):
        logging.info('starting to create encoding and decoding maps')
        try:
            #get all the words and punctuations as seperate words
            word_list=re.findall(r"<EOT>|_|\w+|[\?'\":;$%!&,.()]",txt)
            #remove duplicate words and sort the words (a little inefecient but works for now)
            word_list=sorted(list(set(word_list)))
            encode={}
            count=0
            #encode the words from 0 to n add a token for words that aren't present in the encode hashmap, I will add more tokens for other purposes in the future
            for word in word_list:
                encode[word]=count
                count+=1
            encode['<unknown>']=count
            decode={}
            #change key to value and vice versa
            for k,v in encode.items():
                decode[v]=k
            logging.info('completed encoding text')
            return (encode,decode)

        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
    
    def encode(self,text,key):
        try:
            logging.info('starting encoding the text')
            words=re.findall(r"<EOT>|_|\w+|[\?'\":;$%!&,.()]",text)
            encoded_list=[]
            #encode the words using the hashmap(key) if the word is not present add the encoding for unkown word
            missed=0 #to find the number of words in encoding map it will be easier to know why the model is not performing later. 
            hit=0
            for word in words:
                if word in key:
                    hit+=1
                    encoded_list.append(key[word])
                else:
                    missed+=1
                    encoded_list.append(key['<unknown>'])
            logging.info(f'encodinga sdkjfh text completed\nno. of words encoded: {hit}\nno. of words not in key:{missed}')
            return encoded_list
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)
        
    
    def decode(self,words,key):
        logging.info('Starting decoding of text')
        try:
            #convert numbers to words
            word_list=[]
            for word in words:
                if word in key:
                    word_list.append(key[word])
                else:
                    word_list.append('<unknown>')
            #remove spaces bw puctuations and words
            s1=' '.join(word_list)
            s2=re.sub("\s+([\_,!?;:.()%'])", r"\1",s1)
            logging.info('decoding complete')
            return s2
        
        except Exception as e:
            logging.info('__.__ERROR OCCOURED__.__')
            raise CustomException(e,sys)


if __name__=='__main__':
    text=get_text('/Users/pranavks/Desktop/Projects/transformer/artifacts')
    tk=Tokenizer()
    encode_key,decode_key=tk.transcode(text)
    encoded_text=tk.encode(text,encode_key)
    decode_text=tk.decode(encoded_text,decode_key)
    pos=iter(decode_text)
    print(next(pos))

