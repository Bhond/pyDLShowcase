import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import ToTensor
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas
import os


class YoloDataset(Dataset):
    def __init__(self, config, folderPath, mode='train'):
        """
        Ctor
        :param folderPath: Path to the data folder
        :param mode: Either 'train' or 'test
        """
        self.config = config
        self.folderPath = folderPath
        self.mode = mode
        self.labels = ["aeroplane","bicycle","bird","boat","bottle","bus",
                       "car","cat","chair","cow","diningtable","dog",
                       "horse","motorbike","person","pottedplant","sheep",
                       "sofa","train","tvmonitor"]

    def selectFile(self):
        if self.mode == 'train':
            file = open(self.folderPath + "/train.csv")
        else:
            file = open(self.folderPath + "/test.csv")
        return file

    def pathFromCsv(self,idx):
        """
        Retrieve the image path from the csv summary
        :param idx: Index of the image in the csv
        :return: tuple (imagePath, targetsPath)
        """
        # Open file
        file = self.selectFile()

        imagePath = ""
        targetsPath = ""
        if file is not None:
            if idx == 0:
                data = pandas.read_csv(file,nrows=1,names=["Image","Targets"])
            else:
                data = pandas.read_csv(file,skiprows=idx-1,nrows=1,names=["Image","Targets"])
            imagePath = data["Image"].iloc[0]
            targetsPath = data["Targets"].iloc[0]
            file.close()

        return imagePath,targetsPath

    def loadImage(self,imageFilename, printable=False):
        """
        Retrieve image as a np array
        :param imageFilename: Path to the image
        :return: The image as np array
        """
        if printable:
            image = np.array(Image.open(self.folderPath + "/images/" + imageFilename)
                             .convert("RGB")
                             .resize(size=(self.config.imageSize, self.config.imageSize)))
        else:
            image = np.array(Image.open(self.folderPath + "/images/" + imageFilename)
                             .convert("RGB")
                             .resize(size=(self.config.imageSize, self.config.imageSize)), dtype=np.float32)
            image = np.reshape(image, (3, self.config.imageSize, self.config.imageSize))
        return image

    def loadBoxes(self,boxesFilename):
        """
        Retrieve a target
        :param boxesFilename: Path to the boxes file
        :return: A list with the targets for a given image
        """
        boxes = []
        file = open(self.folderPath + "/labels/" + boxesFilename,"r")
        # Loop over the lines
        for line in file:
            tokens = line.split(' ')
            boxes += [(
                float(tokens[1]),  # x
                float(tokens[2]),  # y
                float(tokens[3]),  # w
                float(tokens[4]),  # h
                int(tokens[0])     # c
            )]

        # Close file
        file.close()
        return boxes

    def printImage(self,indices):
        # Retrieve colors
        # Getting the color map from matplotlib
        colour_map = plt.get_cmap("tab20b")
        # Getting 20 different colors from the color map for 20 different classes
        colors = [colour_map(i) for i in np.linspace(0,1,len(self.labels))]

        for idx in indices:
            # Retrieve values
            imagePath, targetsPaths = self.pathFromCsv(idx)
            image = self.loadImage(imagePath, True)
            boxes = self.loadBoxes(targetsPaths)
            height = image.shape[0]
            width = image.shape[1]

            # Create figure and axes
            fig,ax = plt.subplots(1)

            # Add image to plot
            ax.imshow(image)

            # Loop over all the boxes
            for bbox in boxes:
                # Un-scale values
                boxCenterX = bbox[0] * width
                boxCenterY = bbox[1] * height
                boxWidth = bbox[2] * width
                boxHeight = bbox[3] * height

                # Create box
                rect = patches.Rectangle(
                    (boxCenterX - 0.5 * boxWidth, boxCenterY - 0.5 * boxHeight),
                    boxWidth,
                    boxHeight,
                    linewidth=2,
                    edgecolor=colors[bbox[-1]],
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add class name to the patch
                plt.text(
                    boxCenterX - 0.5 * boxWidth,
                    boxCenterY - 0.5 * boxHeight,
                    s=self.labels[bbox[-1]],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": colors[bbox[-1]],"pad": 0}
                )

                # Add center
                center = patches.Ellipse(
                    (boxCenterX, boxCenterY),
                    5,
                    5,
                    linewidth=1,
                    edgecolor=colors[bbox[-1]],
                    facecolor=colors[bbox[-1]])
                ax.add_patch(center)

                # Add grid
                xStep = width / 7
                yStep = height / 7
                x = width / 7
                y = height / 7
                for i in range(7):
                    vline = patches.Rectangle(
                        (x, 0),
                        1,
                        height,
                        linewidth=1,
                        edgecolor='black',
                        facecolor="none",)
                    x += xStep
                    hline = patches.Rectangle(
                        (0,y),
                        width,
                        1,
                        linewidth=1,
                        edgecolor='black',
                        facecolor="none",)
                    y += yStep
                    ax.add_patch(hline)
                    ax.add_patch(vline)


        # Show plot
        plt.show()

    def __getitem__(self,imageIdx):
        imagePath, boxesPaths = self.pathFromCsv(imageIdx)

        # Image
        imageNp = self.loadImage(imagePath)
        image = torch.from_numpy(imageNp)

        # Labels / Targets
        boxesTuple = self.loadBoxes(boxesPaths)
        boxes = torch.tensor(boxesTuple)
        labelTensor = torch.zeros((self.config.C + 5 * self.config.B, self.config.S, self.config.S))
        for box in boxes:
            # Find cell responsible for detecting the box: (i, j) for channel k
            i = int(self.config.S * box[0])
            j = int(self.config.S * box[1])

            # Select the relevant box inside the output
            idx = -1
            if labelTensor[4, i, j] == 0:
                idx = 4
            elif labelTensor[9, i, j] == 0:
                idx = 9
            else:
                print(boxesPaths)
                self.printImage([imageIdx])

            # If there is no object in neither of the boxes
            # Otherwise the object is not taken into account
            if idx != -1:
                # Coordinates in cell
                xCell = self.config.S * box[0] - i
                yCell = self.config.S * box[1] - j
                # Width and Height in the cell
                wCell = self.config.S * box[2]
                hCell = self.config.S * box[3]
                # Set values in the tensor
                labelTensor[0:4, i, j] = torch.tensor([xCell, yCell, wCell, hCell])
                # Confidence of the cell is set to 1 for selected boxes
                labelTensor[idx, i, j] = 1
                # Select the class
                labelTensor[5 * 2 + int(box[4]), i, j] = 1

        return image, labelTensor

    def __len__(self):
        return len(pd.read_csv(self.selectFile()))
