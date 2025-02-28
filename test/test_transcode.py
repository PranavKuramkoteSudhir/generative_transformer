import pytest
from src.utils import get_text
from src.data_prep.transcode import Tokenizer

"""

assigning dummy data

"""
@pytest.fixture
def path():
    return '/Users/pranavks/Desktop/Projects/transformer/test'

@pytest.fixture
def data():
    return 'I am a boy with nothing to do, I sit around and do nothing.<EOT>'

@pytest.fixture
def encode_key():
    encode_test={
                '<EOT>': 0, 
                 'I': 1, 
                 'a': 2, 
                 'am': 3, 
                 'and': 4, 
                 'around': 5, 
                 'boy': 6,
                 '<unknown>':7
                }
    return encode_test
    
@pytest.fixture
def decode_key():
    decode_test={
                0:'<EOT>',
                1: 'I',
                3: 'am', 
                6: 'boy', 
                7: '<unknown>'
                }
    return decode_test


def test_get_text(path):
    assert get_text(path) == 'I am a boy with nothing to do, I sit around and do nothing.<EOT>'

"""

test if encoder is working correctly

"""
def test_transcode(data):
    tk=Tokenizer()
    encode_test={',': 0,   # at some point convert this to a fixture
                 '.': 1, 
                 '<EOT>': 2, 
                 'I': 3, 
                 'a': 4, 
                 'am': 5, 
                 'and': 6, 
                 'around': 7, 
                 'boy': 8, 
                 'do': 9, 
                 'nothing': 10, 
                 'sit': 11, 
                 'to': 12, 
                 'with': 13, 
                 '<unknown>': 14}
    decode_test={0: ',', 
                 1: '.', 
                 2: '<EOT>', 
                 3: 'I', 
                 4: 'a', 
                 5: 'am', 
                 6: 'and', 
                 7: 'around', 
                 8: 'boy', 
                 9: 'do', 
                 10: 'nothing', 
                 11: 'sit', 
                 12: 'to', 
                 13: 'with', 
                 14: '<unknown>'}
    encode,decode=tk.transcode(data)
    assert encode==encode_test
    assert decode==decode_test

"""

test if the text is getting encoded and decoded properly

"""

def test_encode_decode(data,encode_key,decode_key):
    tk=Tokenizer()
    encoded_text=tk.encode(data,encode_key)
    assert encoded_text==[1, 3, 2, 6, 7, 7, 7, 7, 7, 1, 7, 5, 4, 7, 7, 7, 0]
    decoded_string=tk.decode(encoded_text,decode_key)
    assert decoded_string=='I am <unknown> boy <unknown> <unknown> <unknown> <unknown> <unknown> I <unknown> <unknown> <unknown> <unknown> <unknown> <unknown> <EOT>'



