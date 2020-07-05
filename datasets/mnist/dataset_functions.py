import numpy as np
import gzip
import sys
import matplotlib.pyplot as plt
import time

#imgs are 28 x 28 pixels
img_size = 28

# get current time in milliseconds
millis = lambda: int(round(time.time() * 1000))

def onehot(array):
    ret = np.eye(10)[array].T.astype(np.float64)
    return ret.reshape(ret.shape[0:2])

def streamer(batch_size, training_size = 60000, test_size = 10000, onehot_encoded = True, noise = None, autoencoder = False):
    """
    Stream data as numpy array of vectors and labels as numpy arrays
    """
    start_time = millis()

    #open files as files object
    test_data = gzip.open('./datasets/mnist/test-data.gz','r')
    test_label = gzip.open('./datasets/mnist/test-label.gz','r')
    train_data = gzip.open('./datasets/mnist/train-data.gz','r')
    train_label = gzip.open('./datasets/mnist/train-label.gz','r')

    #skip non relevant information:
    test_data.read(16)
    train_data.read(16)
    test_label.read(8)
    train_label.read(8)

    #load data as np array:
    train_data = train_data.read(img_size * img_size * training_size)
    train_label = train_label.read(training_size)
    test_data = test_data.read(img_size * img_size * test_size)
    test_label = test_label.read(test_size)

    train_data = np.frombuffer(train_data, dtype = np.uint8).astype(np.float64) / 255.
    train_label = np.frombuffer(train_label, dtype = np.uint8).astype(int)
    test_data = np.frombuffer(test_data, dtype = np.uint8).astype(np.float64) / 255.
    test_label = np.frombuffer(test_label, dtype = np.uint8).astype(int)

    #make it a vector:
    train_data = train_data.reshape(training_size, img_size * img_size).T
    test_data = test_data.reshape(test_size, img_size * img_size).T

    train_label = train_label.reshape(1, training_size)
    test_label = test_label.reshape(1, test_size)

    # print(f"Loading the data took {millis()-start_time} milliseconds.")

	# SHUFFLE DATA
    train_idx = np.arange(0, training_size)
    np.random.shuffle(train_idx)

	# ONE HOT ENCODING
    import math
    for i in range(math.ceil(training_size / float(batch_size))):
        # trainig data -> range
        start = batch_size * i
        end   = min(start + batch_size, training_size)

        sample_num = end - start

		# random indices of test data (size: sample_num)
        idx = np.arange(0, test_size)
        np.random.shuffle(idx)
        idx = idx[:sample_num]


        if onehot_encoded == True:
            train_label_encoded = onehot(train_label[:,train_idx[start:end]])
            test_label_encoded  = onehot(test_label[:,idx])
        else:
            train_label_encoded = train_label[:,train_idx[start:end]]
            test_label_encoded  = test_label[:,idx]

        train = [train_data[:,train_idx[start:end]],train_label_encoded]
        test = [test_data[:,idx], test_label_encoded]

        if autoencoder == True:
            train[1] = np.copy(train[0])
            test[1]  = np.copy(test[0])

        if noise != None:
            train[0] += noise * np.random.randn(*train[0].shape)
            test[0]  += noise * np.random.randn(*test[0].shape)

            np.clip(train[0],0.0,1.0)
            np.clip(test[0],0.0,1.0)

        yield (train, test, sample_num)

def show_data_member(data, label, index, predicted_label = "None", oneHot = True, dangerous = False):
    img = data[:,index]
    img = img.reshape(img_size,img_size)
    plt.imshow(img)
    #all different kinds of format don't look just enjoy
    if oneHot == True:
        if predicted_label != "None":
            plt.title(f"    correct label: {np.argmax(label[:,index])}\npredicted label: {np.argmax(predicted_label[:,index])}")
        else:
            plt.title(f"correct label: {np.argmax(label[:,index])}")
    else:
        if predicted_label != "None":
            plt.title(f"    correct label: {label[:,index]}\npredicted label: {predicted_label[:,index]}")
        else:
            plt.title(f"correct label: {label[:,index]}")
    if dangerous == False:
        plt.show(block = True)
    else:
        plt.show(block = False)


if __name__ == '__main__':
    onehot([])
