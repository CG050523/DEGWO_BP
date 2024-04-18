import numpy as np

def func_weight(x, inputnum, hiddennum, outputnum):

    # Extract BP neural network initial weights and thresholds, x for individual
    w1 = x[0:inputnum * hiddennum]
    B1 = x[inputnum * hiddennum:inputnum * hiddennum + hiddennum]
    w2 = x[inputnum * hiddennum + hiddennum:inputnum * hiddennum + hiddennum + hiddennum * outputnum]
    B2 = x[inputnum * hiddennum + hiddennum + hiddennum * outputnum:inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum]

    # Assign network weights
    W_in_hide = np.reshape(w1, (inputnum, hiddennum))
    b1 = np.reshape(B1, (1, hiddennum))
    W_hide_out = np.reshape(w2, (hiddennum, outputnum))
    b2 = np.reshape(B2, (1, outputnum))

    return W_in_hide, b1, W_hide_out, b2