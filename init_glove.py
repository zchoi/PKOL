from genericpath import exists
from json.decoder import JSONDecodeError
import os
import sys
import numpy as np
import collections

import json

from torchtext.data.utils import get_tokenizer

class Caption_vocabulary(object):
    """
     A simple Vocabulary class which maintains a mapping between words and integer tokens. Can be
     initialized either by word counts from the tgif-qa dataset, or a pre-saved vocabulary mapping.

     Parameters
     ----------
     word_counts_path: str
         Path to a json file containing counts of each word across captions, questions and answers
         of the VisDial v1.0 train dataset.
     min_count : int, optional (default=0)
         When initializing the vocabulary from word counts, you can specify a minimum count, and
         every token with a count less than this will be excluded from vocabulary.
    """
    PAD = '<PAD>'
    UNK = '<UNK>'
    SOS = '<S>'
    EOS = '</S>'
    
    PAD_index = 0
    UNK_index = 1
    SOS_index = 2
    EOS_index = 3

    def __init__(self, word_counts_json, file_exist = '') -> None:
        if not exists(word_counts_json):
            raise FileNotFoundError(f'Word counts do not exist at {word_counts_json}')
        if file_exist == '':
            with open(word_counts_json,'r')  as word_counts_file:
                word_counts = json.load(word_counts_file)
            word_counts = [
                (word, count) for word,count in word_counts.items()
            ]
            words = [w[0] for w in word_counts]

            self.word2index = {}
            self.word2index.setdefault(self.PAD,0)
            self.word2index.setdefault(self.UNK,1)
            self.word2index.setdefault(self.SOS,2)
            self.word2index.setdefault(self.EOS,3)

            for index, word in enumerate(words):
                self.word2index.setdefault(word, index + 4)
        else:
            with open(file_exist,'r') as F:
                f = json.load(F)
            self.word2index = f

        self.index2word = {index: word for word,index in self.word2index.items()}

        # json.dump(self.word2index, open('data/tgif-qa/tgif-caption/tgif_word2index.json', 'w'))

    
    @staticmethod
    def _tokenizer(raw_file):
        """
        A simple tokenizer to convert caption sentences to tokens

        Parameters
        ----------
        raw_file: str
            Path to a json file containing name of each video file and its corresponding description.
        
        """

        tokenizer=get_tokenizer('basic_english')
        with open('data/tgif-qa/tgif-caption/tgif-caption.json','r') as f:
            cap_file = json.load(f)  # {gif_name: description}

        word_counts = collections.defaultdict(lambda :0)
        for _, des in cap_file.items():
            token=tokenizer(des)
            for tok in token:
                word_counts[tok] += 1
        
        json.dump(word_counts,open('data/tgif-qa/tgif-caption/tgif-word_counts.json','w'))


class GloveProcessor(object):
  def __init__(self, glove_path):
    self.glove_path = glove_path

  def _load_glove_model(self):
    print("Loading pretrained word vectors...")
    with open(self.glove_path, 'r') as f:
      model = {}
      for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])  # e.g., 300 dimension
        model[word] = embedding

    print("Done.", len(model), " words loaded from %s" % self.glove_path)

    return model

  def save_glove_vectors(self, vocabulary, glove_npy_path, dim=300):
    """
    Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    # vocabulary index2word
    vocab_size = len(vocabulary.index2word)
    glove_embeddings = self._load_glove_model()
    embeddings = np.zeros(shape=[vocab_size, 300], dtype=np.float32)

    vocab_in_glove = 0
    for i in range(0, vocab_size):
      word = vocabulary.index2word[i]
      if word in ['<PAD>', '<S>', '</S>']:
        continue
      if word in glove_embeddings:
        embeddings[i] = glove_embeddings[word]
        vocab_in_glove += 1
      else:
        embeddings[i] = glove_embeddings['unk']

    print("Vocabulary in GLoVE : %d / %d" % (vocab_in_glove, vocab_size))
    np.save(glove_npy_path, embeddings)


if __name__ == '__main__':
    # vocabulary = Caption_vocabulary('data/tgif-qa/tgif-caption/tgif_word_counts.json', file_exist='data/tgif-qa/tgif-caption/tgif_word2index.json')
    # glove_vocab = GloveProcessor('/mnt/hdd1/zhanghaonan/code/MVAN-VisDial-master/glove.6B.300d.txt')
    # glove_vocab.save_glove_vectors(vocabulary, 'data/tgif-qa/tgif-caption/glove.npy')

    with open('data/tgif-qa/tgif-caption/tgif_word2index.json', 'r') as mapping:
        map = json.load(mapping)

    
    
    tokenizer=get_tokenizer('basic_english')
    with open('data/tgif-qa/tgif-caption/tgif_caption.json','r') as f:
        cap_file = json.load(f)  # {gif_name: description}

    res = collections.defaultdict(lambda :[])

    for vid, des in cap_file.items():
        token=tokenizer(des)
        res[vid].append(des)
        w = []
        for tok in token:
            w.append(map.get(tok,1))
        res[vid].append(w)
    json.dump(dict(res),open('data/tgif-qa/tgif-caption/tgif_cap_index.json','w'))