import pytest

from src.data_prep.bpe_transcode import BPETokenizer

@pytest.fixture
def data():
    return "She sells seashells by the seashore. The shells she sells are surely seashells. So if she sells shells on the seashore, I'm sure she sells seashore shells."

def test_train_transcode(data):
    tokenizer=BPETokenizer()

    corpus=data.split(' ')
    corpus_list=[]
    for word in corpus:
        corpus_list.append(list(word))
    
    tokenizer.train(corpus_list)
    encoded_list=[]
    for word in corpus_list:
        encoded_list.extend(tokenizer.encode(word))
    
    assert tokenizer.decode(encoded_list)==data
        