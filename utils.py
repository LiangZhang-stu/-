import numpy as np

def accuracy(predictions, labels):
    err = np.sqrt(np.sum((predictions - labels)**2, 1))
    return np.mean(err), np.var(err, ddof=1)