"""
Class responsible for holding the training and testing
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from math import *

class Trainer:
    def __init__(self,
                 config,
                 model,
                 optimizer,
                 trainingDataset,
                 testingDataset):
        """
        Ctor
        Handles configuring the trainer
        :param config: The project's configuration
        :param trainingDataset: Dataset used for training
        :param testingDataset:  Dataset used for testing
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset

    def run(self):
        """
        Responsible for training the neural network and testing it
        """
        for epoch in range(self.config.numEpochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train()
            self.test()

    def train(self):
        # Dataloader
        dataloader = DataLoader(dataset=self.trainingDataset, batch_size=1)
        size = len(self.trainingDataset)

        # Train mode
        self.model.train()

        # Stop trigger
        idx = 0

        for batch,(X,groundTruth) in enumerate(dataloader):
            # Forward phase
            pred = self.model(X)
            loss = self.computeLoss(pred, groundTruth, 0.5)
            #
            # # Backward
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            #
            # # Verbose
            # if batch % 100 == 0:
            #     loss,current = loss.item(),batch * self.config.batchSize + len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\r")
            #
            # idx += 1
            # if idx == 10:
            #     break

    def test(self):
        # Dataloader
        dataloader = DataLoader(dataset=self.testingDataset)

        # Train mode
        self.model.eval()

    def computeLoss(self, pred, groundTruth, iouThreshold):

        # Retrieve predicted box with best IoU
        predictedBoxes = []
        minIdx = 0

        for i in range(self.config.B):
            predictedBoxes += pred[minIdx:minIdx+4, ...].unsqueeze(0)
            minIdx += 5

        # Box loss

        # Object in cell loss

        # No object in cell loss

        # Class prediction loss


        return .0# F.cross_entropy(pred,groundTruth)
