from abc import ABC, abstractmethod
import numpy as np
from activations import Identity, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Activation
from loss import MSE, CrossEntropy

class Optimizer(ABC):
    """
    An abstract class which represents loss functions which can be applied
    to the output layer in a neural network.
    """
    def __init__(self, learningRate = 0.001):
        self.learningRate = learningRate

    @abstractmethod
    def optimize(self):
        pass

    def setLearningFactor(self, batchSize):
        self.learningFactor = self.learningRate / batchSize

class Adam(Optimizer):

    def __init__(self,learningRate = 0.001, beta_1 = 0.9, beta_2 = 0.999):
        self.learningRate = learningRate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-8
        self.m = None #Initialize 1st momentum
        self.v = None #Initialize 2nd momentum
        self.t = 0

    def optimize(self, variable, variableGradient):
        r"""
        This is a vectorized Adam routine.

        .. math::

            \begin{align*}
                \theta_{t+1} &= \theta_{t} - \frac{\eta \cdot \hat{m_t}}{\sqrt{\hat{v_t}}+\epsilon} \\
                &\text{where} \\
                \hat{m_t} &= \frac{m_t}{1-\beta_1^t} \\
                \hat{v_t} &= \frac{v_t}{1-\beta_2^t} \\
                &\text{and where} \\
                m_t &= (1-\beta_1)g_t + \beta_1 m_{t-1} \\
                v_t &= (1-\beta_2)g_t^2 + \beta_2 v_{t-1}
            \end{align*}

        Epsilon :math:`\epsilon`, which is just a small term preventing division by zero.
        This term is usually :math:`10^{-8}`.
        For the learning rate :math:`\eta` a good default setting is :math:`\eta` = 0.001,
        which is also the default learning rate in Keras.
        The gradient g is :math:`\nabla J(\theta_{t,i})`.

        For further details see https://arxiv.org/pdf/1412.6980v9.pdf for the original paper,
        or https://mlfromscratch.com/optimizers-explained/#/ for a good explanation on optimizers.
        """

        if self.m is None:
            self.m = np.zeros(variableGradient.shape)
            self.v = np.zeros(variableGradient.shape)

        self.t = self.t + 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * variableGradient                #Update biased first moment estimate
        self.v = self.beta_2 * self.v + (1- self.beta_2) * np.square(variableGradient)      #Update biased second raw moment estimate

        m_hat = 1.0/(1.0- self.beta_1**self.t) * self.m                                             #Compute bias-corrected first moment estimate
        v_hat = 1.0/(1.0- self.beta_2**self.t) * self.v                                             #Compute bias-correct second raw moment estimate

        variable -= self.learningFactor * np.divide(m_hat,np.sqrt(v_hat)+ self.epsilon)



class SGD(Optimizer):

    def __init__(self, learningRate=0.01, momentum=0):
        self.learningRate = learningRate
        self.momentum = momentum
        self.variable_update = None

    def optimize(self, variable, variableGradient):
        r"""
        This is a vectorized stochastic gradient descent routine.

        Like before, the activated outputs a and the errors :math:`\delta` are matrices,
        where :math:`\delta^l` equals contains the columns :math:`\delta^{x,l}`, which are
        the errors in layer :math:`l` caused by the NN input vector :math:`x`.
        The learning rate η has an impact, on how quickly the neural network
        learns.

        *Stochastic gradient descent:*

        .. math::

            \begin{align*}
                &\text{for } l \in \{ L, \dots, 1\}\\
                &\qquad \textit{update the weights:} \\
                &\qquad W^l = W^l - \frac{\eta}{|X|} \delta^{l}
                \left(a^{l-1}\right)^T  \\
                &\qquad \textit{update the biases:} \\
                &\qquad b^l = b^l - \frac{\eta}{|X|} \sum_{x \in X}\delta^{x,l}
            \end{align*}

        """
        #initialize the update
        if self.variable_update is None:
            self.variable_update = np.zeros(variable.shape)
        #use momentum to update
        self.variable_update = self.momentum * self.variable_update + (1.0 - self.momentum) * variableGradient
        #update variable
        variable -= self.learningFactor * self.variable_update
