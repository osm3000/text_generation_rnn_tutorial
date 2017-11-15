from sklearn.utils import shuffle
import numpy as np

def getbatch(*args, **kwargs):
    """
    Give it any number of arguments
    """
    i = kwargs['i']
    batch_size = kwargs['batch_size']
    assert len(args) > 0
    output_list = []
    # min_len = min(batch_size, len(args[0]) - 1 - i)
    min_len = min(batch_size, len(args[0]) - i)
    for argument in args:
        output_list.append(argument[i:i + min_len])
    return output_list

def get_random_batch(*args, **kwargs):
    """
    Inspired by Dawood's idea, which seems very logical to me
    """
    args_new = shuffle(args)
    i = kwargs['i']
    batch_size = kwargs['batch_size']
    assert len(args_new) > 0
    output_list = []
    # min_len = min(batch_size, len(args[0]) - 1 - i)
    min_len = min(batch_size, len(args_new[0]) - i)
    for argument in args_new:
        output_list.append(argument[i:i + min_len])
    return output_list

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    # preds = np.log(preds) / temperature
    preds = preds / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
