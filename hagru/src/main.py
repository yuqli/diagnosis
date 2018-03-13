#!/usr/bin/env python
# ==============================================================================
#                                Boilerplate
# ==============================================================================
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
from models import AttentionWordRNN, AttentionSentRNN
import train
from preprocess import find_max_shape

# ==============================================================================
#                               Read Data
# ==============================================================================

def load_pkl(filename):
    '''This function loads a pkl file'''
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

pp_all = ['Sentences-PlainVocab', 'Sentences-Leven', 'Sentences-Byte5K',
       'Sentences-Byte10K', 'Sentences-Byte25K', 'Sentences-Hybrid10K'] # all preprocessing types

ftype = 'Full'
index = 0
pptype = pp_all[index]
filename1 = ftype + '-' + pptype
df = load_pkl(filename1)

# ==============================================================================
#                            Preprocessing Data
# ==============================================================================

X = df[pptype]
Y = df['Labels']

# change Y type
for i in range(len(Y)):
    Y[i] = np.asarray(Y[i], dtype=np.int64)

# change X type
for i in range(len(X)):
    for j in range(len(X[i])):
        X[i][j] = np.asarray(X[i][j], dtype=np.int64)
    X[i] = np.asarray(X[i])

max_sent_len_value, max_token_len_value = find_max_shape(X)

def find_vocabulary_size(dat):
    '''
    The input is a list of lists
    '''
    new_list = []
    for i in range(len(dat)):
        for j in range(len(dat[i])):
            new_list.extend(dat[i][j])
    return int(max(set(new_list))+1)

X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size = 0.3, random_state= 42)


# Model parameters
vocabulary_size_value = find_vocabulary_size(df[pptype]) # for all data, before training and testing split
n_classes_value = len(df.Labels[0])
max_sent_value, max_token_value = find_max_shape(X)

embed_size_value = 50
word_gru_hidden_value = 50
sent_gru_hidden_value = 50
batch_size_value = 64

word_attn = AttentionWordRNN(batch_size=batch_size_value, vocabulary_size=vocabulary_size_value, max_tokens = max_token_value, embed_size=embed_size_value,
                             word_gru_hidden=word_gru_hidden_value, bidirectional= True)

sent_attn = AttentionSentRNN(batch_size=batch_size_value, sent_gru_hidden=sent_gru_hidden_value, max_sents = max_sent_value, word_gru_hidden=word_gru_hidden_value,
                             n_classes=n_classes_value, bidirectional= True)

# Optimization parameters
learning_rate = 1e-1
momentum = 0.9
word_optimizer = torch.optim.SGD(word_attn.parameters(), lr=learning_rate, momentum= momentum)
sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate, momentum= momentum)
criterion = nn.BCELoss()

# training parameters
num_epoch_value = 10

train.train_early_stopping('test'+str(index)+'.txt', max_sent_len_value, max_token_len_value, batch_size_value, X_train, y_train, X_test, y_test, word_attn, sent_attn,
                     word_optimizer, sent_optimizer, criterion, num_epoch_value, 1, 1)
