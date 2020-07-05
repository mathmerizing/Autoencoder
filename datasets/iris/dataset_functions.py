import numpy as np
from sklearn.datasets import load_iris

def streamer(batch_size, training_size = 150, test_size = 150, onehot_encoded = False, noise = None, autoencoder = False):
    test_data = []
    test_label = []

    # load iris
    iris = load_iris()

    X = np.array(iris.data).T
    y = np.array(iris.target).reshape(1,150)

    training_size = X.shape[1]
    test_size = training_size

    import math
    for i in range(math.ceil(training_size / float(batch_size))):
        # trainig data -> range
        start = batch_size * i
        end   = min(start + batch_size, training_size)

        sample_num = end - start

		# random indices of test data (size: sample_num)
        idx = np.random.randint(test_size, size = sample_num)
        data  = [X[:,idx], y[:,idx]]

        if autoencoder == True:
            data[1] = data[0][:]

        """
        if noise != None:
            data[0] += noise * np.random.randn(data[0].shape)
        """

        yield (data, data, sample_num)

def show_data_member():
    pass
