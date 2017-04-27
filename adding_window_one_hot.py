import numpy as np
def adding_window_one_hot(onehot, windowsize):
    """
    Add window slice to a song unit
    :param onehot:
    :param windowsize:
    :return:
    """
    onehot_window_matrix = []
    length = onehot.shape[0]
    for i in range(length):

        if(i >= windowsize):  # add the left window size features
            onehot_window = onehot[i - windowsize]
            for j in range(i-windowsize+1, i+1):
                onehot_window = np.concatenate((onehot_window, onehot[j]))

        else:
            onehot_window = [0] * onehot.shape[1]
            for j in range(windowsize - i - 1):  # add zeros
                onehot_window = np.concatenate((onehot_window, [0] * onehot.shape[1]))
            for j in range(i + 1):  # add the partial left from onehot
                onehot_window = np.concatenate((onehot_window, onehot[j]))
        if(windowsize + i < length):
            for j in range(i+1, i+1+windowsize):
                onehot_window = np.concatenate((onehot_window, onehot[j])) # add the right window size features
        else:
            for j in range(i+1, length):
                onehot_window = np.concatenate((onehot_window, onehot[j]))  # add the partial right window size features
            for j in range(length, i + 1 + windowsize):
                onehot_window = np.concatenate((onehot_window, [0] * onehot.shape[1])) # add zeros
        if (i == 0):
            onehot_window_matrix = np.concatenate((onehot_window_matrix, onehot_window))
        else:

            onehot_window_matrix = np.vstack((onehot_window_matrix, onehot_window))
    return onehot_window_matrix

