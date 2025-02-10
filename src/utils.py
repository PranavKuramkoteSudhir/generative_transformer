import os
from src.logger import logging
from src.exceptions import CustomException
import sys

"""

takes a string dir as input and returns a concatinated text string of all the text files in the provided path

"""
def get_text(file_path):
    try:
        logging.info(f'Reading files from {file_path}')
        dirs=[]
        for rt,fd,fn in os.walk(file_path):
            for file_name in fn:
                extension= file_name.split('.')[1]
                if extension=='txt':
                    dirs.append(os.path.join(rt,file_name))
        text=''
        for path in dirs:
            with open(path, "r", encoding="utf-8") as file:
                text+=file.read()
            text+='<EOT>'
        logging.info(f'text extraction completed from {file_path}. \nno. of file found: {len(dirs)}\nsample text: {text[:50]}')
        return text
    except Exception as e:
        logging.info('__.__ ERROR OCCOURED__.__')
        raise CustomException(e,sys)
if __name__=='__main__':
    get_text("/Users/pranavks/Desktop/Projects/transformer/artifacts")
            




