import torch


class Config:
    """
    Class responsible for holding the configuration for the whole project
    """

    def __init__(self):
        self.numEpochs = 1
        self.batchSize = 100
        self.learningRate = 1e-2
        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.imageSize = 448
        self.S = 7
        self.B = 2
        self.C = 20
        self.lamdaCoord = 1
        self.lamdaObj = 1
        self.lamdaNoObj = .5
