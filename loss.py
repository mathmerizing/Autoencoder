from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    """
    An abstract class which represents loss functions which can be applied
    to the output layer in a neural network.
    """

    @abstractmethod
    def apply(self,x):
        """
        Evaluate loss function.
        """
        pass

    @abstractmethod
    def derivative(self,x):
        """
        Evaluate derivative of loss function.
        """
        pass

class MSE(Loss):
    """
    Mean Squared Error (MSE)
    """

    def apply(self,y,a):
        r"""
        .. math::
            \text{Loss}(X;W,b) := \frac{1}{2|X|} \sum_{x \in X} ||NN(x;W,b) - y(x)||^2

        with the input batch X, the weights W, the biases b,
        y(x) the desired output and NN(x;W,b) the output from the NN
        when x is inputted.
        """
        temp = a - y
        return (1. / (2. * y.shape[1])) * np.sum(temp * temp)

    def derivative(self,y,a):
        r"""
        .. math::
            \nabla_{NN} \text{Loss}(X;W,b) = \frac{1}{|X|} \sum_{x \in X} \left( NN(x;W,b) - y(x) \right)
        """
        return (1. / y.shape[1]) * np.sum(a - y, axis=1).reshape((y.shape[0],1))


class CrossEntropy(Loss):
    """
    Categorical Crossentropy
    """
    def apply(self,y,a):
        #reference https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60#3ee0
        a = np.clip(a,1e-10,1-1e-10)    #preventing division by zero
        return (-1. / y.shape[1]) * np.sum(y*np.log(a) +(np.ones(y.shape)-y)*np.log(np.ones(a.shape)-a))

    def derivative(self,y,a):
        r"""
        This is the derivative of the CrossEntropy loss with a softmax layer as input.
        In https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax,
        it has been derived that

        .. math::

           \nabla_{z} \text{Loss}(X;W,b) = \frac{1}{|X|} \left( z(X;W,b) - y(x) \right)

        where z is the output of the last layer of the neural network, before it has been
        activated with the Softmax function.
        """
        a = np.clip(a,1e-10,1-1e-10)    #preventing division by zero
        return (1. / y.shape[1]) * (np.divide(-y,a)+np.divide(np.ones(y.shape)-y,np.ones(a.shape)-a))
