import numpy as np

def loader(training_size = 1000, test_size = 1000, training_Batch_number = 0, test_Batch_number = 0):
    test_data = []
    test_label = []

    for line in open('./datasets/xor/input.txt').readlines():
        for i in line.split('\n')[0].split(' '):
            test_data.append(int(i))

    for line in open('./datasets/xor/output.txt').readlines():
        test_label.append(int(line.split('\n')[0]))

    test_label = np.array(test_label).reshape(1,1000)
    test_data = np.array(test_data)
    test_data = test_data.reshape(2,1000)
    train_data = test_data
    train_label = test_label
    return[train_data, train_label, test_data, test_label]

def streamer(batch_size, training_size = 1000, test_size = 1000, onehot_encoded = True):
    test_data = []
    test_label = []

    for line in open('./datasets/xor/input.txt').readlines():
        for i in line.split('\n')[0].split(' '):
            test_data.append(int(i))

    for line in open('./datasets/xor/output.txt').readlines():
        test_label.append(int(line.split('\n')[0]))

    test_label = np.array(test_label).reshape(1,1000)
    test_data = np.array(test_data)
    test_data = test_data.reshape(1000,2).T
    train_data = test_data
    train_label = test_label

    import math
    for i in range(math.ceil(training_size / float(batch_size))):
        # trainig data -> range
        start = batch_size * i
        end   = min(start + batch_size, training_size)

        sample_num = end - start

		# random indices of test data (size: sample_num)
        idx = np.random.randint(test_size, size = sample_num)

        train = [train_data[:,start:end], train_label[:,start:end]]
        test  = [test_data[:,idx], test_label[:,idx]]
        yield (train, test, sample_num)

def show_data_member(data, label, index, predicted_label = "None", oneHot = True, dangerous = False):
    #data: 2x1000   label: 1x1000
    print(data[:,index],label[0,index])
