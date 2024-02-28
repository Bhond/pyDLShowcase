"""
Class responsible for holding the training and testing
"""


class Trainer:
    def __init__(self, config, model):
        """
        Ctor
        Handles configuring the trainer
        :param config: The project's configuration
        """

    def run(self):
        """
        Responsible for training the neural network and testing it
        """
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train()
            self.test()

    def train(self):
       print("Train")

    def test(self):
        print("Test")
