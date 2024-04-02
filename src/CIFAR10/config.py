"""
Class responsible for holding the project's configuration
"""
import torch


class Config:
    def __init__(self):
        self.num_epochs = 5
        self.batch_size = 64
        self.learning_rate = .1
        self.data_path = "../data/"
        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
