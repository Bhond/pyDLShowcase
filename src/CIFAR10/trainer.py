"""
Class responsible for holding the training and testing
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.CIFAR10.dataset import CIFAR10Dataset


class Trainer:
    def __init__(self, config, model):
        """
        Ctor
        Handles configuring the trainer
        :param config: The project's configuration
        """
        self.config = config
        self.model = model
        self.training_dataset = CIFAR10Dataset(config, True)
        self.testing_dataset = CIFAR10Dataset(config, False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    def run(self):
        """
        Responsible for training the neural network and testing it
        """
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train()
            self.test()

    def train(self):
        """
        Training method
        """
        dataloader = DataLoader(self.training_dataset, self.config.batch_size)
        size = len(self.training_dataset)
        loss = 0.0
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Forward pass
            pred, loss = self.model.forward(X, y)
            # Backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * self.config.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\r")

    def test(self):
        """
        Testing method
        """
        dataloader = DataLoader(self.testing_dataset, self.config.batch_size)
        size = len(self.testing_dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        # Set the model to evaluation mode, check tutorial for full description of what this line does
        self.model.eval()
        # Start loop: retrieves the data, the label
        with torch.no_grad():
            for X, y in dataloader:
                # Forward pass
                pred, loss = self.model.forward(X, y)
                # Loss
                test_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                # Compute score and display it
                test_loss /= num_batches
                correct /= size
                print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
