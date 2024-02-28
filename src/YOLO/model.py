from torch import nn


class NeuralNetwork(nn.Module):
    """
    Class responsible for holding the neural network used to learn the features
    """

    def __init__(self):
        """
        Ctor
        """
        # Parent
        super(NeuralNetwork, self).__init__()

    def forward(self, x):
        print("Forward method")

