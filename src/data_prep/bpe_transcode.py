import re
import collections

"""

This is just a demonstration for the understanding of BPE tokenization, I will end up using the tiktokenizer while working on the model anyways

"""

class BPETokenizer:
    """

    inetialize all the dictionaries needed
         
    """
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.merges = {} # Stores merge rules
        self.vocab = {} # Maps subwords to integer IDs
        self.vocab_reverse={} # Maps integers to subword IDs

    """

    Count the frequency of token pairs in the text corpus.

    """
    def get_stats(self, corpus):
        pairs = collections.defaultdict(int)
        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        return pairs

    """
    
    Merge the most frequent pair into a new token.
    
    """

    def merge_vocab(self, pair, corpus):

        new_corpus = []
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)') # ensures that there isnt a space before and after the pair

        for word in corpus:
            word_str = ' '.join(word)
            merged_word_str = pattern.sub(''.join(pair), word_str)
            new_word = merged_word_str.split()
            new_corpus.append(new_word)

        return new_corpus

    """

    Train the tokenizer using BPE until vocabulary size is reached.
    
    """

    def train(self, text_corpus):

        # Append "</w>" to to show the difference bw patterns at the beginning and ending of a word
        corpus = []
        for word in text_corpus:
            corpus.append(list(word) + ["</w>"])  # Append </w> to each word

        # Initialize the vocabulary with unique characters + </w>
        unique_tokens = set()
        for word in corpus:
            for token in word:
                unique_tokens.add(token)

        while len(unique_tokens) < self.vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs:
                break  # No more pairs to merge

            # Find the most frequent pair
            best_pair = None
            highest_count = 0
            for pair, count in pairs.items():
                if count > highest_count:
                    best_pair = pair
                    highest_count = count

            if best_pair is None:
                break

            corpus = self.merge_vocab(best_pair, corpus)
            merged_token = ''.join(best_pair)

            # Stop if vocab size is reached
            unique_tokens.add(merged_token)
            if len(unique_tokens) >= self.vocab_size:
                break

            self.merges[best_pair] = merged_token

        # Assign integer token IDs
        self.vocab = {}
        index = 0
        for token in unique_tokens:
            self.vocab[token] = index
            index += 1
        self.vocab_reverse = {}
        for token, idx in self.vocab.items():
            self.vocab_reverse[idx] = token

    """
        
    Encode a word into integer token IDs using trained BPE merges.

    """

    def encode(self, word):

        if not self.merges:
            raise ValueError("Tokenizer not trained yet. Run `train(corpus)` first.")

        word = list(word) + ["</w>"]

        while True:
            # Generate all adjacent pairs
            pairs = []
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs.append(pair)

            # Filter only pairs that exist in merge rules
            valid_pairs = []
            for pair in pairs:
                if pair in self.merges:
                    valid_pairs.append(pair)

            if not valid_pairs:
                break  # No more merges possible

            # Find the best pair to merge (basically its best to go through the vocabulary in the same order as the training so I end up with the same word after encoding))
            best_pair = valid_pairs[0]
            for pair in valid_pairs:
                if self.merges[pair] < self.merges[best_pair]:
                    best_pair = pair

            # Merge the best pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(self.merges[best_pair])  # Merge the pair
                    i += 2  # Skip the next token since it's merged
                else:
                    new_word.append(word[i])  # Keep the token as is
                    i += 1

            word = new_word  # Update the word with merged subwords

        # Convert subwords to integer token IDs
        token_ids = []
        for token in word:
            token_ids.append(self.vocab[token])

        return token_ids

    """
    
    Decode a list of integer token IDs back into a word (removing </w>).
    
    """

    def decode(self, token_ids):
        decoded_word = ""
        for token_id in token_ids:
            token = self.vocab_reverse[token_id]
            decoded_word += token
        punctuations=set(['.',',','/','?','!','@','&','(',')','[',']'])
        word_list=decoded_word.split('</w>')
        s=''
        for word in word_list:
            if word in punctuations:
                s+=word
            else:
                s=s+' '+word
            s=s.strip()
        return s

