"""
Class responsible for dealing with the CIFAR10 dataset
"""
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

from config import Config


class CIFAR10Dataset(Dataset):
    def __init__(self, config, transforms=None):
        """
        Ctor
        :param config: Project's configuration
        :param transforms: The transform used for data augmentation
        """
        # Augmentation
        self.transforms = transforms

        # Data
        self.data = datasets.CIFAR10(
            root=config.data_path,
            train=config.train,
            transform=ToTensor(),
            download=True
        )

    def print(self, image):
        """
        Print an image with matplotlib
        :param image: The image to print
        """
        # TODO: Add label
        plt.figure(figsize=(2, 2))
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

    def __get_item__(self, idx):
        """
        Override
        Allow data augmentation
        :param idx: Data's index in the dataset
        :return: The data and its label
        """
        img, label = self.data.__getitem__(idx)
        if self.transforms:
            return self.transforms(img), label

    def __len__(self):
        """
        Override
        :return: dataset's length
        """
        self.data.__len__()
