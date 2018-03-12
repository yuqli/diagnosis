import numpy as np
import time
import math
from sklearn import metrics
import torch
from torch.autograd import Variable

def test_evaluation_mini_batch(tokens, labels, word_attn_model, sent_attn_model):
    """
    This function returns the macro-f score for tokens, labels
    This is the harmonic mean of average precision and recall over all nodes.
    """
    y_pred = get_predictions(tokens, word_attn_model, sent_attn_model)
    y_pred = torch.round(y_pred)
    precision = 0
    recall = 0
    for i in range(y_pred.size()[1]):
        p = y_pred[ :, 1]
        l = labels[ :, 1]
        p = np.ndarray.flatten(p.data.cpu().numpy())
        l = np.ndarray.flatten(l.data.cpu().numpy())
        precision += metrics.precision_score(l, p)
        recall += metrics.recall_score(l, p)
    precision_avg = precision/float(y_pred.size()[1])
    recall_avg = recall/float(y_pred.size()[1])
    macro_f = 2.0 / ((1.0/(precision_avg+0.001)) + (1.0 / (recall_avg + 0.001)))
    return macro_f

def test_data(mini_batch, targets, word_attn_model, sent_attn_model, loss_criterion):
    state_word = word_attn_model.init_hidden()
    state_sent = sent_attn_model.init_hidden()
    max_sents, batch_size, max_tokens = mini_batch.size()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        _s = _s.view(1, batch_size, -1)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    y_pred, state_sent = sent_attn_model(s, state_sent)
    loss = loss_criterion(y_pred, targets)
    return loss.data[0]

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def gen_minibatch(tokens, labels, mini_batch_size, max_sent_len, max_token_len, shuffle= True):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token = pad_batch(token, max_sent_len, max_token_len)
        label = pad_batch_y(label).view(mini_batch_size, -1)
        yield token, label

def check_val_loss(val_tokens, val_labels, mini_batch_size, max_sent_len, max_token_len, word_attn_model, sent_attn_model, loss_criterion):
    val_loss = []
    for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
        val_loss.append(test_data(pad_batch(token, max_sent_len, max_token_len), pad_batch_y(label).view(mini_batch_size, -1), word_attn_model, sent_attn_model, loss_criterion))
    return np.mean(val_loss)

def train_data(mini_batch, targets, word_attn_model, sent_attn_model, word_optimizer, sent_optimizer, criterion):
    state_word = word_attn_model.init_hidden()
    state_sent = sent_attn_model.init_hidden()
    max_sents, batch_size, max_tokens = mini_batch.size()
    word_optimizer.zero_grad()
    sent_optimizer.zero_grad()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        _s = _s.view(1, batch_size, -1)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    y_pred, state_sent = sent_attn_model(s, state_sent)
    loss = criterion(y_pred, targets)
    loss.backward()

    word_optimizer.step()
    sent_optimizer.step()

    return loss.data[0]

# This is to pad each mini_batch into the same size of max_sent_len and max_token
def pad_batch(mini_batch, max_sent_len, max_token_len):
    mini_batch_size = len(mini_batch)
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i,j,k] = mini_batch[i][j][k]
                except IndexError:
                    main_matrix[i, j, k] = 1
    return Variable(torch.from_numpy(main_matrix).transpose(0,1))

# This is to pad each label mini-batch
def pad_batch_y(mini_batch):
    mini_batch_size = len(mini_batch)
    num_labels = len(mini_batch[0])
    main_matrix = np.zeros((mini_batch_size, num_labels), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
                try:
                    main_matrix[i,j] = mini_batch[i][j]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix).transpose(0,1)).float()

def get_predictions(val_tokens, word_attn_model, sent_attn_model):
    max_sents, batch_size, max_tokens = val_tokens.size()
    state_word = word_attn_model.init_hidden()
    state_sent = sent_attn_model.init_hidden()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(val_tokens[i,:,:].transpose(0,1), state_word)
        _s = _s.view(1, batch_size, -1)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    y_pred, state_sent = sent_attn_model(s, state_sent)
    return y_pred

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_early_stopping(out_name, max_sent_len, max_token_len, mini_batch_size, X_train, y_train, X_test, y_test, word_attn_model, sent_attn_model,
                         word_attn_optimiser, sent_attn_optimiser, loss_criterion, num_epoch,
                         print_val_loss_every = 1, print_loss_every = 1):
    start = time.time()
    loss_full = []
    loss_epoch = []
    macro_f_epoch = []
    macro_f_full = []
    epoch_counter = 0
    g = gen_minibatch(X_train, y_train, mini_batch_size, max_sent_len, max_token_len)
    f = open(out_name,'w')
    for i in range(1, num_epoch + 1):
        try:
            tokens, labels = next(g)
            loss = train_data(tokens, labels, word_attn_model, sent_attn_model, word_attn_optimiser,
                              sent_attn_optimiser, loss_criterion)
            macro_f = test_evaluation_mini_batch(tokens, labels, word_attn_model, sent_attn_model)
            macro_f_epoch.append(macro_f)
            macro_f_full.append(macro_f)
            loss_full.append(loss)
            loss_epoch.append(loss)
            # print loss every n passes
            if i % print_loss_every == 0:
                f.write('Loss at %d minibatches, %d epoch,(%s) is %f \n' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch)))
                f.write('Macro F at %d minibatches is %f \n' % (i, np.mean(macro_f_epoch)))
            # check validation loss every n passes
            if i % print_val_loss_every == 0:
                val_loss = check_val_loss(X_train, y_train, mini_batch_size, max_sent_len, max_token_len, word_attn_model,
                                          sent_attn_model, loss_criterion)
                f.write('Average training loss at this epoch..minibatch..%d..is %f \n' % (i, np.mean(loss_epoch)))
                f.write('Validation loss after %d passes is %f \n' %(i, val_loss))
                if val_loss > np.mean(loss_full):
                   f.write('Validation loss is higher than training loss at %d is %f , stopping training! \n' % (i, val_loss))
                    f.write('Average training loss at %d is %f \n' % (i, np.mean(loss_full)))
        except StopIteration:
            epoch_counter += 1
            f.write('Reached %d epocs \n' % epoch_counter)
            f.write('i %d \n' % i)
            g = gen_minibatch(X_train, y_train, mini_batch_size, max_sent_len, max_token_len)
            loss_epoch = []
    f.write('The full loss is : \n' + str(loss_full))
    f.close()
    return loss_full
