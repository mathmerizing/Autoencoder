import numpy as np
import numpy.matlib as matlib
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import json
import copy

from nn import MLP
from activations import Identity, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Activation
from loss import MSE, CrossEntropy
from tensorboardX import SummaryWriter
from dataset import Dataset
from optimizer import Adam, SGD
from layers import Dense
from my_utils import prettyTime

class Sampler():

    def __init__(self, inputDim = 1 , outputDim = 1, optimizer = Adam()):
        self.inputDim  = inputDim
        self.outputDim = outputDim
        self.mean   = Dense(self.inputDim, self.outputDim, activation = Identity() , optimizer = copy.copy(optimizer))
        self.logVar = Dense(self.inputDim, self.outputDim, activation = Identity() , optimizer = copy.copy(optimizer))

    def feedforward(self, input):
        self.latentMean   = self.mean.feedforward(input)
        self.latentLogVar = self.logVar.feedforward(input)

        self.epsilon = np.random.standard_normal(size = (self.outputDim,input.shape[1]))
        self.sample  = self.latentMean + np.exp(self.latentLogVar / 2.) * self.epsilon

        return self.sample

    def backpropagate(self, lastGradient, timeStep = 1):
        gradLogVar = {}
        gradMean   = {}
        tmp        = self.outputDim * lastGradient.shape[1]

        # KL divergence gradients
        gradLogVar["KL"] = (np.exp(self.latentLogVar) - 1) / (2 * tmp)
        gradMean["KL"]   = self.latentMean / tmp

        # MSE gradients
        gradLogVar["MSE"] = 0.5 * lastGradient * self.epsilon * np.exp(self.latentLogVar / 2.)
        gradMean["MSE"]   = lastGradient

        # backpropagate gradients thorugh self.mean and self.logVar
        return self.mean.backward(gradMean["KL"] + gradMean["MSE"], timeStep = timeStep) + self.logVar.backward(gradLogVar["KL"] + gradLogVar["MSE"], timeStep = timeStep)

    def getKLDivergence(self, output):
        # output.shape[1] == batchSize
        return - np.sum(1 + self.latentLogVar - np.square(self.latentMean) - np.exp(self.latentLogVar)) / (2 * self.outputDim * output.shape[1])

class VAE(MLP):

    def __init__(self, encoder = None, sampler = None, decoder = None):
        super().__init__()

        if encoder != None and sampler != None and decoder != None:
            self.layers  = encoder.layers + [sampler.mean, sampler.logVar] +  decoder.layers
            self.encoder = encoder
            self.sampler = sampler
            self.decoder = decoder
            self.decoder.loss = MSE()

    def feedforward(self, input):
        encoderOutput = self.encoder.feedforward(input)
        sample = self.sampler.feedforward(encoderOutput)
        decoderOutput = self.decoder.feedforward(sample)

        return decoderOutput

    def backpropagate(self, output, timeStep = 1):
        self.decoder.backpropagate(output, timeStep = timeStep)
        decoderGradient = self.decoder.layers[0].gradient
        samplerGradient = self.sampler.backpropagate(decoderGradient, timeStep = timeStep)
        self.encoder.backpropagate(samplerGradient, timeStep = timeStep, useLoss = False)

    def train(self, dataset, loss = MSE(), epochs = 1, metrics = ["train_loss", "test_loss"], tensorboard = False, callbacks = {}):
        super().train(dataset, loss = loss, epochs = epochs, metrics = metrics, tensorboard = tensorboard, callbacks = callbacks, autoencoder = True, noise = None)

    def getLoss(self,output):
        return self.decoder.getLoss(output) + self.sampler.getKLDivergence(output)

    def __str__(self):
        out  = "-" * 20 + " VARIATIONAL AUTOENCODER (VAE) " + "-" * 20 + "\n\n"
        out += f"TOTAL PARAMETERS = {sum(l.numParameters() for l in self.layers)} \n\n"

        out += "#" * 15 + "\n"
        out += "#   ENCODER   #\n"
        out += "#" * 15 + "\n\n"
        for i, layer in enumerate(self.encoder.layers):
            out += f" *** {i+1}. Layer: *** \n"
            out += str(layer) + "\n"

        out += "#" * 15 + "\n"
        out += "#   SAMPLER   #\n"
        out += "#" * 15 + "\n\n"
        out += f" *** MEAN Layer: *** \n"
        out += str(self.sampler.mean) + "\n"
        out += f" *** LOG_VAR Layer: *** \n"
        out += str(self.sampler.logVar) + "\n"

        out += "#" * 15 + "\n"
        out += "#   DECODER   #\n"
        out += "#" * 15 + "\n\n"
        for i, layer in enumerate(self.decoder.layers):
            out += f" *** {i+1}. Layer: *** \n"
            out += str(layer) + "\n"

        out += "-" * 70 + "\n"
        return out

    def load(self,name):
        modelDir = f"./models/{name}"

        self.encoder = MLP()
        self.decoder = MLP()
        self.decoder.loss = MSE()

        # load encoder and decoder
        for name, model in [("encoder", self.encoder), ("decoder", self.decoder)]:
            layerDir = [dir for dir in os.listdir(modelDir) if os.path.isdir(os.path.join(modelDir, dir)) and name in dir]
            layerDir.sort(key = lambda x : int(x.strip(f"{name}_layer")))

            for dir in layerDir:
                layerFolder = os.path.join(modelDir, dir)
                if "dense.json" in os.listdir(layerFolder):
                    # this is a dense layer
                    newLayer = Dense()
                    newLayer.load(layerFolder)
                    model.layers.append(newLayer)

        # load aditional information about sampler
        with open(f"{modelDir}/sampler.json", "r") as file:
            data = json.load(file)

        inputDim  = data["inputDim"]
        outputDim = data["outputDim"]
        self.sampler = Sampler(inputDim, outputDim)

        # load mean and logvar layer
        self.sampler.mean = Dense()
        self.sampler.mean.load(os.path.join(modelDir, f"sampler_mean"))
        self.sampler.logVar = Dense()
        self.sampler.logVar.load(os.path.join(modelDir, f"sampler_logvar"))

        self.layers = self.encoder.layers + [self.sampler.mean, self.sampler.logVar] +  self.decoder.layers

    def save(self,name):
        # save: weights, biases --> with NUMPY
        modelDir = f"./models/{name}"
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)

        # save encoder
        for i, layer in enumerate(self.encoder.layers):
            layer.save(f"{modelDir}/encoder_layer{i}")

        # save sampler
        self.sampler.mean.save(f"{modelDir}/sampler_mean")
        self.sampler.logVar.save(f"{modelDir}/sampler_logvar")

        # save supplementary data about sampler
        data = {}
        data["inputDim"] = self.sampler.inputDim
        data["outputDim"] = self.sampler.outputDim
        with open(f"{modelDir}/sampler.json", "w") as file:
            json.dump(data, file)

        # save decoder
        for i, layer in enumerate(self.decoder.layers):
            layer.save(f"{modelDir}/decoder_layer{i}")
