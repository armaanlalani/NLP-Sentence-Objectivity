"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from models import *
import readline

def main():
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    dataset = data.TabularDataset(path='data/data.tsv', format='tsv', skip_header=True, fields=[('text',TEXT), ('label',LABELS)])

    # 3.2.4
    TEXT.build_vocab(dataset)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    baseline = torch.load('model_baseline.pt')
    cnn = torch.load('model_cnn.pt')
    rnn = torch.load('model_rnn.pt')

    while True:
        print("Enter a sentence")
        text = input()
        tokens = tokenizer(text)
        token_ints = [vocab.stoi[tok] for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1,1)
        lengths = torch.Tensor([len(token_ints)])
        base_prob = baseline(token_tensor, lengths, 1)
        cnn_prob = cnn(token_tensor, lengths, 1)
        rnn_prob = rnn(token_tensor, lengths, 1)
        if base_prob > 0.5:
            print("Model baseline: subjective (%.3f)" %(base_prob))
        else:
            print("Model baseline: objective (%.3f)" %(1-base_prob))
        if cnn_prob > 0.5:
            print("Model cnn: subjective (%.3f)" %(cnn_prob))
        else:
            print("Model cnn: objective (%.3f)" %(1-cnn_prob))
        if rnn_prob > 0.5:
            print("Model rnn: subjective (%.3f)" %(rnn_prob))
        else:
            print("Model rnn: objective (%.3f)" %(1-rnn_prob))
        

def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

if __name__ == '__main__':
    main()
