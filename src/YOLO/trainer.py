"""
Class responsible for holding the training and testing
"""
import torch


class Trainer:
    def __init__(self, config, model):
        """
        Ctor
        Handles configuring the trainer
        :param config: The project's configuration
        """
        self.config = config
        self.model = model

    def run(self):
        """
        Responsible for training the neural network and testing it
        """
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train()
            self.test()

    def train(self):
        test = torch.rand(3, 448, 448)
        self.model.forward(test)

    def test(self):
        print("Test")
