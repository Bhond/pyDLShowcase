import torch


class Config:
    """
    Class responsible for holding the configuration for the whole project
    """

    def __init__(self):
        self.num_epochs = 1
        self.batch_size = 64
        self.learning_rate = 0.1
        self.data_path = "../data/"
        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
