"""
Entry file for the CIFAR10 project
"""
from dataset import *
from model import *
from trainer import *
from config import *

def main():
    """
    Main method responsible for dispatching the work:
        - Create the configuration
        - Create the dataset
        - Create the model
        - Create the trainer
        - Run the trainer
    """
    config = Config()
    cifar10Dataset = CIFAR10Dataset(config)
    model = NeuralNetwork(config)
    trainer = Trainer(config)


if __name__ == "__main__":
    """
    Entry point
    """
    main()
