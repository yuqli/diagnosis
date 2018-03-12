import numpy as np

def find_max_shape(data):
    max_sent_len = int(np.mean([len(x) for x in data]))
    max_token_len = int(np.mean([len(val) for sublist in data for val in sublist]))
    return max_sent_len, max_token_len


