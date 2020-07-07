import numpy as np
import numpy.matlib as matlib
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import json
import copy

from nn import MLP, Writer
from activations import Identity, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Activation
from loss import MSE, CrossEntropy
from tensorboardX import SummaryWriter
from dataset import Dataset
from optimizer import Adam, SGD
from layers import Dense
from my_utils import prettyTime

"""
Look at https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
for more informations
"""
class GAN(MLP):

    def __init__(self, generator = MLP(), discriminator = MLP()):
        super().__init__()

        if generator != None and discriminator != None:
            self.generator = generator
            self.discriminator = discriminator
            self.layers  = self.generator.layers + self.discriminator.layers
            self.generator.loss = CrossEntropy()
            self.discriminator.loss = CrossEntropy()

    def __str__(self):
        out  = "-" * 20 + " GENERATIVE ADVERSARIAL NETWORK (GAN) " + "-" * 20 + "\n\n"
        out += f"TOTAL PARAMETERS = {sum(l.numParameters() for l in self.layers)} \n\n"

        out += "#" * 17 + "\n"
        out += "#   GENERATOR   #\n"
        out += "#" * 17 + "\n\n"
        for i, layer in enumerate(self.generator.layers):
            out += f" *** {i+1}. Layer: *** \n"
            out += str(layer) + "\n"

        out += "#" * 21 + "\n"
        out += "#   DISCRIMINATOR   #\n"
        out += "#" * 21 + "\n\n"
        for i, layer in enumerate(self.discriminator.layers):
            out += f" *** {i+1}. Layer: *** \n"
            out += str(layer) + "\n"

        out += "-" * 70 + "\n"
        return out

    def sample(self,batchSize):
        return np.random.standard_normal(size = (self.generator.layers[0].inputDim,batchSize))

    def train(self, dataset, loss = MSE(), epochs = 1, metrics = ["generator_loss", "discriminator_loss"], tensorboard = False, callbacks = {}, autoencoder = False, noise = None):
        metricsWriter = Writer(metrics, callbacks, tensorboard)

        imgNoise = np.random.standard_normal(size = (self.generator.layers[0].inputDim,10))

        ind = 0 # number of samples processed
        for i in range(epochs):
            logging.debug(f" *** EPOCH {i+1}/{epochs} ***")
            for (train, test, batchSize) in dataset.batches(onehot_encoded = True, autoencoder = autoencoder, noise = noise):
                # set batch size before training
                for layer in self.layers:
                    layer.setBatchSize(batchSize)

                # 1. Train discriminator
                fake_img = self.generator.feedforward(self.sample(batchSize))
                real_img = np.asarray(train[0])

                input = np.concatenate((fake_img,real_img), axis = 1)
                label = np.concatenate(
                            (
                                np.zeros((1,batchSize)),
                                0.9 * np.ones((1,batchSize))
                            ),
                            axis = 1
                        )

                self.discriminator.feedforward(np.asarray(input))
                self.discriminator.backpropagate(np.asarray(label))
                discriminatorLoss =  self.discriminator.getLoss(np.asarray(label))

                # 2. Train generator
                #fake_img = self.generator.feedforward(self.sample(batchSize))
                self.discriminator.feedforward(fake_img)
                self.discriminator.backpropagate(np.ones((1,batchSize)), updateParameters = False)
                discriminatorGradient = self.discriminator.layers[0].gradient
                self.generator.backpropagate(discriminatorGradient, useLoss = False)
                generatorLoss = self.discriminator.getLoss(np.ones((1,batchSize)))

                if ind % 1000 < batchSize:
                    if "generator_loss" in metrics:
                        metricsWriter.add(metric = "generator_loss",     index = ind, value = generatorLoss)
                    if "discriminator_loss" in metrics:
                        metricsWriter.add(metric = "discriminator_loss", index = ind, value = discriminatorLoss)

                    #self.validate(test, ind, callbacks, writer = metricsWriter, metrics = metrics)

                ind += batchSize
            self.generateImages(imgNoise,i)
        metricsWriter.close()

    def generateImages(self,noise,epoch):
        generatedImgs = self.generator.feedforward(noise)
        if np.__name__ == "cupy":
            generatedImgs = np.asnumpy(generatedImgs)
        plt.figure(figsize=(10, 10))
        plt.title(f"Epoch {epoch}")

        for i in range(10):
            plt.subplot(10, 10, i+1)
            plt.imshow(generatedImgs[:,i].reshape((28, 28)), cmap='gray')
            plt.axis('off')
        plt.savefig(f"ganImages/{epoch}.png")


    def save(self,name):
        # save: weights, biases --> with NUMPY
        modelDir = f"./models/{name}"
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)

        # save generator
        for i, layer in enumerate(self.generator.layers):
            layer.save(f"{modelDir}/generator_layer{i}")

        # save discriminator
        for i, layer in enumerate(self.discriminator.layers):
            layer.save(f"{modelDir}/discriminator_layer{i}")

    def load(self,name):
        modelDir = f"./models/{name}"

        # load generator and discriminator
        for name, model in [("generator", self.generator), ("discriminator", self.discriminator)]:
            layerDir = [dir for dir in os.listdir(modelDir) if os.path.isdir(os.path.join(modelDir, dir)) and name in dir]
            layerDir.sort(key = lambda x : int(x.strip(f"{name}_layer")))

            for dir in layerDir:
                layerFolder = os.path.join(modelDir, dir)
                if "dense.json" in os.listdir(layerFolder):
                    # this is a dense layer
                    newLayer = Dense()
                    newLayer.load(layerFolder)
                    model.layers.append(newLayer)

        self.layers = self.generator.layers  +  self.discriminator.layers



if __name__ == "__main__":
    dataset = Dataset(name = "mnist", train_size = 60000, test_size = 10000, batch_size = 128)
    LATENT_SIZE = 28*28
    # set the learning rate and optimizer for training
    optimizer = Adam(0.0002,0.5)

    generator = MLP()
    generator.addLayer(Dense(inputDim = LATENT_SIZE, outputDim = 256, activation = LeakyReLU(0.2), optimizer = optimizer))
    generator.addLayer(Dense(inputDim = 256, outputDim = 512, activation = LeakyReLU(0.2), optimizer = optimizer))
    generator.addLayer(Dense(inputDim = 512, outputDim = 1024, activation = LeakyReLU(0.2), optimizer = optimizer))
    generator.addLayer(Dense(inputDim = 1024, outputDim = 28*28, activation = Tanh(), optimizer = optimizer))

    discriminator = MLP()
    discriminator.addLayer(Dense(inputDim = 28*28, outputDim = 1024, activation = LeakyReLU(0.2), optimizer = optimizer))
    discriminator.addLayer(Dense(inputDim = 1024, outputDim = 512, activation = LeakyReLU(0.2), optimizer = optimizer))
    discriminator.addLayer(Dense(inputDim = 512, outputDim = 256, activation = LeakyReLU(0.2), optimizer = optimizer))
    discriminator.addLayer(Dense(inputDim = 256, outputDim = 1, activation = Sigmoid(), optimizer = optimizer))

    gan = GAN(generator,discriminator)
    print(gan)

    gan.train(dataset,loss = MSE(), epochs = 50, metrics = ["generator_loss", "discriminator_loss"], tensorboard = True, callbacks = [])

    gan.save("tryout_gan")
