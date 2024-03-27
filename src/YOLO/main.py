from config import *
from model import *
from trainer import *


def main():
    config = Config()
    model = YoloV1(config)
    model.build()
    trainer = Trainer(config, model)
    trainer.run()


if __name__ == "__main__":
    main()
