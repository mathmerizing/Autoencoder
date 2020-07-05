from pySmartDL import SmartDL
import os
import numpy as np
import sys

class Dataset():
    """
    Dataset is the class to load and interact with different Datasets.
    Once a dataset is loaded, it stores train_data, train_label, test_data, test_label and train_size and test_size.
    Formatting the data is done when loading a specific dataset and is handeled by datasets/dataset/dataset_functions.py
    Dataset also loads a specific print method defined in datasets/dataset/dataset_functions.py
    Download information is centrally stored in datasets/sources.txt and can be modified by this classes download method.
    """
    def __init__(self, name, train_size, test_size, batch_size, redownload = False):
        #creates Dataset instance
        self.name = name
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        if redownload == False:
            if self.name not in self.showDownloadedDatasets():
                self.downloadDataset(name, redownload = True)
        else:
            self.downloadDataset(name, redownload = True)

    def showDatasetSources(self):
        """
        Shows dataset sources saved in /datasets/sources.txt
        """
        print("Datasets sourced in /datasets/sources.txt:")
        with open("./datasets/sources.txt",'r') as file:
            lines = file.readlines()
            #skips over header line
            for i in range(2,len(lines)):
                if lines[i-1] == '\n':
                    #print dataset name without new line character
                    print(">>>", lines[i].split('\n')[0])

    def showDownloadedDatasets(self):
        """Shows currently downloaded datasets"""
        Downloaded_datasets = []
        for i in os.listdir('./datasets/'):
            if i != "sources.txt":
                #look inside the folders
                for j in os.listdir('./datasets/' + i):
                    if "dataset_functions" not in j and "pycache" not in j and i not in Downloaded_datasets:
                        Downloaded_datasets.append(i)
        return Downloaded_datasets


    def get_source(self, name):
        """
        Loads source links and file names from /datasets/sources.txt
        """
        self.source =[]
        #flag to check if dataset was found
        found = False
        #scans for dataset name line by line
        with open("./datasets/sources.txt",'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if found == True:
                    #
                    if '\n' == lines[i]:
                        break
                    else:
                        #removes new line character at end of line and saves [file name, url]
                        self.source.append(lines[i].split('\n')[0].split(' '))
                if name +'\n' == lines[i]:
                    #look for dataset name
                    print('found dataset')
                    found = True
        #returns if dataset was found
        return found

    def downloadDataset(self, name, redownload = False):
        """
        Downloads a dataset. If it is unknown, it will ask the user to specify datanames and links for downloading.
        """
        #check if dataset is already downloaded
        if not os.path.isdir("./datasets/" + name) or redownload:
            #download predetermined dataset
            if self.get_source(name):
                pass

            #download custom dataset
            else:
                print('We do not have a source for this dataset, please enter a source below')
                while True:
                    file_name = input("please enter a filename (enter nothing to end): ")
                    #break on empty input
                    if not file_name:
                        break
                    file_link = input("please enter a link: ")
                    self.source.append([file_name, file_link])

                #save new source to /datasets/sources.txt
                #add new line character if it is missing.
                if open("./datasets/sources.txt",'r').readlines()[-1] !='\n':
                    open("./datasets/sources.txt",'a').write('\n')
                with open("./datasets/sources.txt",'a') as file:
                    file.write(name + '\n')
                    for i in self.source:
                        file.write(i[0] + ' ' + i[1] + '\n')
                    file.write('\n')

            #download files
            for i in self.source:
                file = SmartDL(i[1], "./datasets/" + name + "/" +i[0])
                file.start()
        #if folder for the dataset already exists
        else:
            print("Dataset "+ name + " already exists, if you'd like to redownload it, recall this downloadDataset and set redownload to True!")

    def batches(self, onehot_encoded = True, noise = None, autoencoder = False):
        """
        Streams a specific dataset
        """
        sys.path.append("./datasets/" + self.name +"/")
        from dataset_functions import streamer, show_data_member
        self.print = show_data_member

        for batch in streamer(self.batch_size, self.train_size, self.test_size, onehot_encoded = onehot_encoded, noise = noise, autoencoder = autoencoder):
            #self.print(batch[0][0],batch[0][1], 0)
            yield batch
