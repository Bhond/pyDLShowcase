"""
Class responsible for holding the project's configuration
"""


class Config:
    def __init__(self):
        self.num_epochs = 5
        self.batch_size = 64
        self.learning_rate = 0.1
        self.train = True;
        self.data_path = "../data/"