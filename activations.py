from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Activation(ABC):
    """
    An abstract class which represents activation functions which can be applied
    to a layer in a neural network.
    """

    _name = "activation name"

    @abstractmethod
    def apply(self,x):
        """
        Evaluate activation function.
        """
        pass

    @abstractmethod
    def derivative(self,x):
        """
        Evaluate derivative of activation function.
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def funcFromStr(name):
        functions = {
            "Identity": Identity(),
            "Sigmoid":  Sigmoid(),
            "Softmax":  Softmax(),
            "ReLU":     ReLU(),
            "Tanh":     Tanh(),
        }
        if "LeakyReLU" not in name:
            return functions[name]
        else:
            # loading LeakyReLU
            _, eps = name.split("_")
            return LeakyReLU(float(eps))


    def _plot(self):
        """
        Plot the graph of the activation function.
        """
        x = np.linspace(-10, 10, 100)
        plt.plot(x, self.apply(x))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(self._name)
        #plt.savefig(self._name.split()[0] + ".svg", transparent = True)
        plt.show()

class Identity(Activation):
    """
    Identity function
    """
    _name = "Identity function"

    def apply(self,x):
        r"""
        .. math::
            f(x) := x
        """
        return x

    def derivative(self,x):
        r"""
        .. math::
            f'(x) := 1
        """
        return np.ones(x.shape)

    def __str__(self):
        return "Identity"

class Sigmoid(Activation):
    """
    .. image:: img/Sigmoid.svg
        :alt: sigmoid
        :align: center
    """
    _name = "Sigmoid function"

    def apply(self,x):
        r"""
        .. math::
            \sigma(x) := \frac{1}{1 + \exp(-x)}
        """
        return 1. / (1. + np.exp(-x))

    def derivative(self,x):
        r"""
        .. math::
            \sigma '(x) :=\sigma(x) \cdot (1 - \sigma(x))
        """
        return self.apply(x) * (1. - self.apply(x))

    def __str__(self):
        return "Sigmoid"


class Tanh(Activation):
    """
    .. image:: img/Tanh.svg
        :alt: tanh
        :align: center
    """
    _name = "Tanh - Hyperbolic tangent"

    def apply(self,x):
        r"""
        .. math::
            \tanh(x) := \frac{\sinh(x)}{\cosh(x)}
        """
        return np.tanh(x)

    def derivative(self,x):
        r"""
        .. math::
            \tanh'(x) := 1 - \tanh^2(x)
        """
        return 1. - np.power(np.tanh(x),2)

    def __str__(self):
        return "Tanh"


class ReLU(Activation):
    """
    .. image:: img/ReLU.svg
        :alt: relu
        :align: center
    """
    _name = "ReLU - Rectified linear unit"

    def apply(self,x):
        r"""
        .. math::
            \text{relu}(x) := \max(0,x)
        """
        return x * (x > 0)

    def derivative(self,x):
        r"""
        .. math::
            \text{relu}'(x) :=
            \begin{cases}
                0 & \text{for } x \leq 0 \\
                1 & \text{else} \\
            \end{cases}
        """
        return 1. * (x > 0)

    def __str__(self):
        return "ReLU"

class LeakyReLU(Activation):
    """
    .. image:: img/LeakyReLU.svg
        :alt: leaky_relu
        :align: center
    """
    _name = "LeakyReLU - Leaky rectified linear unit"
    epsilon = 0

    def __init__(self,epsilon=0.01):
        self.epsilon = epsilon
        super().__init__()

    def apply(self,x):
        r"""
        .. math::
            \text{leaky_relu}(x) := \max(\varepsilon x,x) \text{ with } \varepsilon \ll 1
        """
        return np.where(x > 0, x, x * self.epsilon)

    def derivative(self,x):
        r"""
        .. math::
            \text{leaky_relu}'(x) :=
            \begin{cases}
                \varepsilon & \text{for } x \leq 0 \\
                1 & \text{else} \\
            \end{cases}
        """
        return np.where(x > 0, 1., self.epsilon)

    def __str__(self):
        return f"LeakyReLU_{self.epsilon}"

class Softmax(Activation):
    """
    Softmax function
    """
    _name = "Softmax function"

    def apply(self,x):
        r"""
        .. math::
            \text{softmax}(x_i) := \frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)}
        """
        tmp = np.exp(x - np.max(x, axis=0)) # - np.max(x) prevents under-/overflow
        return tmp / tmp.sum(axis=0)

    def derivative(self,x):
        #reference https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        return self.apply(x) * (1. - self.apply(x)) 

    def __str__(self):
        return "Softmax"


if __name__ == "__main__":
    for func in [Identity(),Sigmoid(),Tanh(),ReLU(),LeakyReLU(),Softmax()]:
        func._plot()
