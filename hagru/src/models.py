import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class AttentionWordRNN(nn.Module):
    def __init__(self, batch_size, vocabulary_size, max_tokens, embed_size, word_gru_hidden, bidirectional=True):

        super(AttentionWordRNN, self).__init__()

        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.max_tokens = max_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        self.lookup = nn.Embedding(vocabulary_size, embed_size)

        if bidirectional == True:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=True)
            self.attn = nn.Linear(self.word_gru_hidden * 2, self.max_tokens)
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=False)
            self.attn = nn.Linear(self.word_gru_hidden, self.max_tokens)

    def forward(self, input_tokens, state_word):
        # embeddings
        embedded = self.lookup(input_tokens)
        embedded = embedded.view(-1, self.batch_size, self.embed_size)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
        # only attend to the first word
        attn_weights = F.softmax(self.attn(output_word[0]))
        # reshape attn_weights for batch multiplication
        attn_weights = attn_weights.view(self.batch_size, 1, -1)
        word_attn_vectors = torch.bmm(attn_weights, output_word.view(self.batch_size, self.max_tokens, -1)).view(
            self.batch_size, -1)
        return word_attn_vectors, state_word, attn_weights.view(self.batch_size, -1)

    # next function
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))


class AttentionSentRNN(nn.Module):
    """Modified: attention_layers = n_classes"""
    """Now test n_classes = 5"""

    def __init__(self, batch_size, max_sents, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True):

        super(AttentionSentRNN, self).__init__()

        self.batch_size = batch_size
        self.max_sents = max_sents
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.attn_list = []

        self.final_linear_list = []

        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
            for i in range(n_classes):
                self.attn_list.append(nn.Linear(sent_gru_hidden * 2, max_sents))
                self.final_linear_list.append(
                    nn.Linear(2 * sent_gru_hidden, 1))  # here bineary classification for each class
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=True)
            for i in range(n_classes):
                self.attn_list.append(nn.Linear(sent_gru_hidden, max_sents))
                self.final_linear_list.append(nn.Linear(sent_gru_hidden, 1))

    def forward(self, word_attention_vectors, state_sent):
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)
        # only implements local attention to the first position
        p = None  # prediction
        for i in range(self.n_classes):
            attn_weights = F.softmax(self.attn_list[i](output_sent[0]))
            attn_weights = attn_weights.view(self.batch_size, 1, -1)
            sent_attn_vectors = torch.bmm(attn_weights, output_sent.view(self.batch_size, self.max_sents, -1)).view(
                self.batch_size, -1)
            final_map = self.final_linear_list[i](sent_attn_vectors)
            m = nn.Sigmoid()
            final_map = m(final_map)
            final_map = final_map.view(self.batch_size, -1)
            if (p is None):
                p = final_map
            else:
                p = torch.cat((p, final_map), 1)
        # final classifier
        return p, state_sent

    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))
