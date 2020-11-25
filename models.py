import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None, batch_size=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)
        output = F.sigmoid(output)

	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return output

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.conv1 = nn.Conv2d(1,n_filters,kernel_size=(filter_sizes[0],embedding_dim)) # first convolutional layer with width 2
        self.conv2 = nn.Conv2d(1,n_filters,kernel_size=(filter_sizes[1],embedding_dim)) # second convolutional layer with width 4
        self.fc = nn.Linear(embedding_dim,1) # linear layer to use at the end

    def forward(self, x, lengths=None, batch_size=None):
        embedded = self.embedding(x)
        embedded = torch.transpose(embedded,0,1) # rearranges the tensor for convolutional layers
        embedded = embedded.unsqueeze(1) # adds a dimenion for the 1 channel
        x_1 = F.relu(self.conv1(embedded)) # first convolutional layer
        x_2 = F.relu(self.conv2(embedded)) # second convolutional layer
        x_1, inds = torch.max(x_1, dim=2) # max pooling of the output of the first convolutional layer
        x_2, inds = torch.max(x_2, dim=2) # max pooling of the output of the second convolutional layer
        x_1 = x_1.squeeze(2) # removes the unnecessary dimention
        x_2 = x_2.squeeze(2) # removes the unnecessary dimension
        x = torch.stack((x_1, x_2),dim=2) # stacks the two outputs on top of each other
        x = torch.reshape(x,(batch_size,100)) # reshapes the stack so that it is batch size x embedded dimension
        output = F.sigmoid(self.fc(x).squeeze(1)) # squeezes the tensor and puts it through the linear layer
        return output

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim) # GRU
        self.fc = nn.Linear(embedding_dim,1) # linear layer

    def forward(self, x, lengths=None, batch_size=None):
        embedded = self.embedding(x)
        embedded = torch.transpose(embedded,0,1) # rearranges tensor for the GRU
        x = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True) # packs the batch based on the lengths of the sentence
        x_unpack, len_unpack = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x, states = self.gru(x_unpack) # runs the padded sequence through the GRU
        x = torch.transpose(x,0,1) # rearranges the matrix
        average = x.mean(0)
        output = self.fc(average).squeeze(1) # runs the output through the linear layer
        output = F.sigmoid(output)
        return output