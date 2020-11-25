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

def main(args):
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    overfit_data = data.TabularDataset(path='data/overfit.tsv', format='tsv', skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
	sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    overfit_iter = data.BucketIterator(overfit_data, batch_size=50, sort_key=lambda x:len(x.text), device=None, sort_within_batch=True, repeat=False)

    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)

    if(args.model == 'baseline'):
        model = Baseline(args.emb_dim, vocab) # instantiates baseline model
    elif(args.model == 'rnn'):
        model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim) # instantiates RNN model
    elif(args.model == 'cnn'):
        model = CNN(args.emb_dim, vocab, args.num_filt, [2,4]) # instantiates CNN model

    loss = torch.nn.BCEWithLogitsLoss() # sets loss to BCE with logits loss
    opt = torch.optim.Adam(model.parameters(), args.lr) # sets optimizer to Adam

    train_accuracy = [] # stores training accuracy after each evaluation step
    train_loss = [] # stores training loss after each evaluation step
    val_accuracy = [] # stores validation accuracy after each evaluation step
    val_loss = [] # stores validation loss after each evaluation step
    total_loss = 0 # temporary variable to hold overall training loss for each batch
    train_acc = 0 # temporary variable to hold overall training accuracy for each batch

    test = False # variable that holds whether to run an overfitting test or regular training loop

    if test:
        for epoch in range(args.epochs):
            for i in range(len(overfit_iter)):
                batch = next(iter(overfit_iter)) # obtains the next batch
                batch_input, batch_input_length = batch.text

                opt.zero_grad()
                outputs = model(batch_input, batch_input_length, 1) # gets the outputs

                labels = batch.label.type(torch.float) # converts labels to floats

                loss_in = loss(input=outputs, target=labels) # computes the loss
                loss_in.backward()
                opt.step()

                train_accuracy.append(evaluate(outputs, labels, 50)) # updates accuracy and loss arrays accordingly
                train_loss.append(loss_in.item())

    if not test:
        for epoch in range(args.epochs):
            for i in range(len(train_iter)):
                batch = next(iter(train_iter)) # obtains the next batch
                batch_input, batch_input_length = batch.text

                opt.zero_grad()
                outputs = model(batch_input, batch_input_length, args.batch_size) # gets the outputs

                labels = batch.label.type(torch.float) # converts labels to floats

                loss_in = loss(input=outputs, target=labels) # computes the loss
                loss_in.backward()
                opt.step()

                total_loss += loss_in.item() # updates temporary loss variable for each batch
                train_acc += evaluate(outputs, labels, args.batch_size) # updates temporary accuracy variable for each batch

                if i % args.eval_every == 0: # evaluation step
                    train_accuracy.append(train_acc/args.eval_every) # update training accuracy based on how often evaluation occurs
                    train_loss.append(total_loss/args.eval_every) # update training loss based on how often evaluation occurs
                    val_l = 0 # temporary variables similar to total_loss and train_acc except for validation
                    val_acc = 0
                    for j in range(len(val_iter)):
                        val = next(iter(val_iter)) # gets the next batch of validation data
                        val_input, val_input_length = val.text
                        val_output = model(val_input, val_input_length, args.batch_size) # gets the outputs for the batch
                        val_labels = val.label.type(torch.float)
                        val_l += loss(input=val_output, target=val_labels) # computes the loss of the batch
                        val_acc += evaluate(val_output, val_labels, args.batch_size) # computes the accuracy of the batch
                    val_accuracy.append(val_acc/len(val_iter)) # updates validation accuracy based on the number of batches
                    val_loss.append(val_l/len(val_iter)) # updates validation loss based on the number of batches
                    print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i, total_loss / args.eval_every))
                    print("Training Accuracy: %.5f" %(train_acc/args.eval_every))
                    print("Validation Accuracy: %.5f" %(val_acc/len(val_iter))) # prints out some specs for the training loop
                    train_acc = 0
                    total_loss = 0

        test_accuracy = 0
        for i in range(len(test_iter)): # loop that calculates the overall test accuracy, similar to above train and validation loops
            batch = next(iter(test_iter))
            batch_input, batch_input_length = batch.text
            labels = batch.label.type(torch.float)

            outputs = model(batch_input, batch_input_length, args.batch_size)

            test_accuracy += evaluate(outputs, labels, args.batch_size)
            if i == len(test_iter)-1:
                loss_in = loss(input=outputs, target=labels) # computes the loss
                loss = loss_in.item()
    
        test_accuracy = test_accuracy / len(test_iter)
        print("Test Accuracy: %.5f" %(test_accuracy))
        print("Test Loss: %.5f" %(loss))

        # torch.save(model, 'model_rnn.pt')

    if not test:
        epochs = np.linspace(0,args.epochs,2*args.epochs) # spaces between every half an epoch because evaluation occurs twice per epoch
    else:
        epochs = np.linspace(0,args.epochs,args.epochs) # when testing, evaluation occurs every epoch

    plt.plot(epochs, train_accuracy, label="Training") # plot of accuracy
    if not test:
        plt.plot(epochs, val_accuracy, label="Validation")
    plt.legend()
    plt.title("Accuracy vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(epochs, train_loss, label="Training") # plot of loss
    if not test:
        plt.plot(epochs, val_loss, label="Validation")
    plt.legend()
    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def evaluate(outputs, labels, length):
    output = outputs.detach().numpy() # changes the output to a numpy array
    label = labels.detach().numpy() # changes the labels to a numpy array
    correct = 0
    for i in range(output.shape[0]): # determines how many outputs are correctly predicted
        if 1 >= output[i] >= 0.5 and label[i] == 1:
            correct += 1
        elif 0 <= output[i] < 0.5 and label[i] == 0:
            correct += 1
    return correct/length # returns the accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=50)

    args = parser.parse_args()

    main(args)