import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import json

from activations import Identity, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Activation
from loss import MSE, CrossEntropy
from tensorboardX import SummaryWriter
from dataset import Dataset
from optimizer import Adam, SGD
from my_utils import prettyTime
from layers import Dense

class Writer():
    def __init__(self, metrics = ["train_loss", "test_loss"], callbacks = {}, tensorboard = False):
        self.metrics = metrics
        self.tensorboard = tensorboard
        self.callbacks = callbacks

        if self.tensorboard:
            # create SummaryWriter which will create a Tensorboard-readable file
            self.summaryWriter = SummaryWriter()

        for metric in self.metrics:
            with open(f"{metric}.txt", "w") as f:
                f.write(f"{metric}\n")

    def add(self, metric, index, value):
        assert metric in self.metrics, "Metric cannot be written."
        # log to txt file
        with open(f"{metric}.txt", "a") as f:
            f.write(f"{index},{value}\n")

        # save to tensorboard
        if self.tensorboard:
            self.summaryWriter.add_scalar(metric, value, index)

        if metric in self.callbacks:
            self.callbacks[metric](index,value)

    def close(self):
        if self.tensorboard:
            self.summaryWriter.close()

class MLP():
    """
    .. image:: img/MLP.svg
        :alt: mlp
        :align: center
    """
    def __init__(self):
        self.layers    = []
        self.loss      = None

    def addLayer(self, layer):
        """
        Add a new layer to the NN.
        """
        self.layers.append(layer)

    def feedforward(self, input):
        r"""
        This is a vectorized forward propagation routine.

        Instead of a single input vector, the MLP receives a mini batch
        of input vectors and identifies them as columns of the matrix.
        Consequently we work with a bias matrix :math:`\tilde{b}`, where each column is
        the same bias vector :math:`b`.
        The activation function σ is being applied column wise.

        *Forward propagation:*

        .. math::

            \begin{align*}
                &a^0 = \text{input} \\
                &\text{for } l \in \{ 1, \dots, L\}\\
                &\qquad z^l = W^l a^{l-1} + \tilde{b}^l \\
                &\qquad a^l = \sigma(z^l)
            \end{align*}

        """
        for layer in self.layers:
            input = layer.feedforward(input)

        return input

    def predict(self, input):
        """
        Forward propgate the input through the NN and
        output the activations of the last layer.
        """
        input  = input.reshape((input.shape[0], 1))
        return self.feedforward(input)

    def backpropagate(self, output,timeStep = 1, useLoss = True, updateParameters = True):
        r"""
        This is a vectorized backpropagation routine.

        Since we work with mini bacthes (see: feedforward),
        the gradient of the loss function and the derivative of the activation
        function σ are being applied to the columns of the matrices.

        *Backpropagation:*

        1. Compute the output error:

        .. math::

            \delta^L = \nabla_{NN} \text{Loss} \odot \sigma'(z^L)

        2. Backpropagate:

        .. math::

            \begin{align*}
                &\text{for } l \in \{ L-1, \dots, 1\}\\
                &\qquad \delta^l = \left((W^l)^T \delta^{l+1} \right) \odot \sigma'(z^l)
            \end{align*}

        """
        if useLoss:
            # step 1:
            lastGradient = self.loss.derivative(output, self.layers[-1].a) * self.layers[-1].activation.derivative(self.layers[-1].z)
            # step 2:
            isOutputLayer = True
            for layer in self.layers[::-1]:
                lastGradient = layer.backward(lastGradient,timeStep = timeStep, outputLayer = isOutputLayer, updateParameters = updateParameters )
                isOutputLayer = False

        else:
            isOutputLayer = False
            lastGradient  = output
            for layer in self.layers[::-1]:
                lastGradient = layer.backward(lastGradient,timeStep = timeStep, outputLayer = isOutputLayer, updateParameters = updateParameters)


    def train(self, dataset, loss = MSE(), epochs = 1, metrics = ["train_loss", "test_loss"], tensorboard = False, callbacks = {}, autoencoder = False, noise = None):
        metricsWriter = Writer(metrics, callbacks, tensorboard)
        self.loss = loss

        ind = 0 # number of samples processed
        for i in range(epochs):
            logging.debug(f" *** EPOCH {i+1}/{epochs} ***")
            for (train, test, batchSize) in dataset.batches(onehot_encoded = True, autoencoder = autoencoder, noise = noise):
                # set batch size before training
                for layer in self.layers:
                    layer.setBatchSize(batchSize)

                self.feedforward(train[0])
                self.backpropagate(train[1],timeStep = i+1)

                if ind % 1000 < batchSize:
                    if "train_loss" in metrics:
                        metricsWriter.add(metric = "train_loss",     index = ind, value = self.getLoss(train[1]))
                    if "train_accuracy" in metrics:
                        metricsWriter.add(metric = "train_accuracy", index = ind, value = self.getAccuracy(train[1]))

                    self.validate(test, ind, callbacks, writer = metricsWriter, metrics = metrics)

                ind += batchSize
        metricsWriter.close()

    def validate(self, test, ind, callbacks, writer = None, metrics = ["train_loss", "test_loss"]):
        self.feedforward(test[0])
        if writer != None:
            if "test_loss" in metrics:
                writer.add(metric = "test_loss",     index = ind, value = self.getLoss(test[1]))
            if "test_accuracy" in metrics:
                writer.add(metric = "test_accuracy", index = ind, value = self.getAccuracy(test[1]))

    def getLoss(self, label):
        return self.loss.apply(label,self.layers[-1].a)

    def getAccuracy(self, label):
        difference = np.argmax(self.layers[-1].a, axis = 0) - np.argmax(label, axis = 0)
        accuracy = (1 - np.count_nonzero(difference) / len(difference)) * 100
        return accuracy

    def getGraph(self):
        """
        Compute the graph object representing the neural network.
        """
        for layer in self.layers:
            assert isinstance(layer, Dense), "Can't compute graph"

        neurons = [self.layers[0].inputDim]
        for layer in self.layers:
            neurons.append(layer.outputDim)

        # create a dictionary which saves nodes in the given layers
        nodes = {}
        for i in range(len(self.layers)+1):
            start = sum(neurons[:i])
            nodes[i] = range(start,start+neurons[i])

        # create a directed Graph
        graph = nx.DiGraph()

        # create edges between consecutive layers
        for l in range(len(self.layers)):
            for x in nodes[l]:
                for y in nodes[l+1]:
                    graph.add_edge(x,y)

        # compute positions of nodes
        maxNodes = max(neurons)
        for layer in range(len(self.layers)+1):
            layerNodes = neurons[layer]
            for i, node in enumerate(nodes[layer]):
                height = i + 0.5 * (maxNodes - layerNodes)
                # save coordinates of node in graph
                graph.nodes[node]['pos'] = (
                    layer,
                    height
                )

        pos = nx.get_node_attributes(graph,'pos')

        #color the nodes
        colorMap = []
        for node in graph.nodes():
            if node in nodes[0]:
                colorMap.append('red')
            elif node in nodes[len(self.layers)]:
                colorMap.append('green')
            else:
                colorMap.append('blue')

        return (graph, pos, colorMap)

    def plotGraph(self, title = "Multi Layer Perceptron (MLP)"):
        """
        Plot the graph of the network's architecure.
        """
        graph, pos, colorMap = self.getGraph()

        fig = plt.figure()
        fig.canvas.set_window_title("Neural Network")
        plt.plot()
        nx.draw_networkx_nodes(graph,pos, node_color = colorMap)
        nx.draw_networkx_edges(graph,pos)
        plt.axis('off')
        plt.title(title)
        #plt.savefig("autoencoder.svg", transparent = True)
        plt.show()

    def getGraphFigure(self, title = "Multi Layer Perceptron (MLP)"):
        """
        Return a Matplotlib figure of the graph of the network's architecture.
        """
        graph, pos, colorMap = self.getGraph()

        fig = plt.figure()
        plt.plot()
        nx.draw_networkx_nodes(graph,pos, node_color = colorMap)
        nx.draw_networkx_edges(graph,pos)
        plt.axis('off')
        plt.title(title)
        return fig

    def __str__(self):
        out  = "-" * 20 + " MULTI LAYER PERCEPTRON (MLP) " + "-" * 20 + "\n\n"
        out += f"HIDDEN LAYERS = {len(self.layers) - 2} \n"
        out += f"TOTAL PARAMETERS = {sum(l.numParameters() for l in self.layers)} \n\n"
        for i, layer in enumerate(self.layers):
            out += f" *** {i+1}. Layer: *** \n"
            out += str(layer) + "\n"
        out += "-" * 70 + "\n"
        return out

    def load(self,name):
        modelDir = f"./models/{name}"
        layerDir = [dir for dir in os.listdir(modelDir) if os.path.isdir(os.path.join(modelDir, dir))]
        layerDir.sort(key = lambda x : int(x.strip("layer")))

        for dir in layerDir:
            layerFolder = os.path.join(modelDir, dir)
            if "dense.json" in os.listdir(layerFolder):
                # this is a dense layer
                newLayer = Dense()
                newLayer.load(layerFolder)
                self.layers.append(newLayer)

    def save(self,name):
        # save: weights, biases --> with NUMPY
        modelDir = f"./models/{name}"
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)

        for i, layer in enumerate(self.layers):
            layer.save(f"{modelDir}/layer{i}")
