"""
Entry file for the CIFAR10 project
"""
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
    model = NeuralNetwork(config)
    trainer = Trainer(config, model)
    trainer.run()


if __name__ == "__main__":
    """
    Entry point
    """
    main()
