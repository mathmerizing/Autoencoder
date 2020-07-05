import numpy as np
import numpy.matlib as matlib
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import json

from nn import MLP
from activations import Identity, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Activation
from loss import MSE, CrossEntropy
from tensorboardX import SummaryWriter
from dataset import Dataset
from optimizer import Adam, SGD
from my_utils import prettyTime

class Autoencoder(MLP):
    def __init__(self, encoder = MLP(), decoder = MLP(), noise = None):
        """
        An Autoencoder consists of an Encoder network and a a Decoder network.
        In the constructor merged these two networks.
        """
        super().__init__()
        self.layers += encoder.layers + decoder.layers
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise

    def predict(self, input):
        """
        Forward propagate the input through the Encoder and
        output the activations of the last layer of the Encoder,
        i.e. return the latent vector.
        """
        return encoder.predict(input)


    def train(self, dataset, loss = MSE(), epochs = 1, metrics = ["train_loss", "test_loss"], tensorboard = False, callbacks = {}):
        super().train(dataset, loss = loss, epochs = epochs, metrics = metrics, tensorboard = tensorboard, callbacks = callbacks, autoencoder = True, noise = self.noise)
