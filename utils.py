from sklearn.utils import shuffle
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
