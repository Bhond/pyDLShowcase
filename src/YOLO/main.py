from config import *
from model import *
from trainer import *
from dataset import *


def main():
    config = Config()
    trainingDataset = YoloDataset(config, "../data/yoloDB")
    testingDataset = YoloDataset(config, "../data/yoloDB", mode='test')
    #trainingDataset.printImage((463, 464, 465, 590, 591, 592))
    model = YoloV1(config)
    model.build()
    optimizer = torch.optim.SGD(model.parameters(),lr=config.learningRate)
    trainer = Trainer(config, model, optimizer, trainingDataset, testingDataset)
    trainer.run()


if __name__ == "__main__":
    main()
