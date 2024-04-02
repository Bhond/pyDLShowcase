"""
Class responsible for holding the neural network used to learn the features
"""
import torch
import torch.nn as nn
from torchsummary import summary


class YoloVn(nn.Module):
    def __init__(self):
        super(YoloVn,self).__init__()


class YoloV1(YoloVn):
    cfg = [
        (7,64,2,3),
        'M',
        (3,192,1,1),
        'M',
        (1,128,1,0),
        (3,256,1,1),
        (1,256,1,0),
        (3,512,1,1),
        'M',
        [(1,256,1,0),
         (3,512,1,1),4],
        (1,512,1,0),
        (3,1024,1,1),
        'M',
        [(1,512,1,0),
         (3,1024,1,1),2],
        (3,1024,1,1),
        (3,1024,2,1),
        (3,1024,1,1),
        (3,1024,1,1)
    ]

    def __init__(self,config):
        """
        Ctor
        :param config: Project's configuration
        """
        # Parent
        super(YoloV1,self).__init__()
        self.fl0 = None
        self.dropout = None
        self.leaky = None
        self.fl1 = None
        self.model = None
        self.config = config

    def build(self):
        in_channels = 3
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            elif isinstance(v,tuple):
                if len(v) == 4:
                    layers += [nn.Conv2d(in_channels=in_channels,
                                         out_channels=v[1],
                                         kernel_size=(v[0]),
                                         stride=v[2],
                                         padding=v[3])]
                else:
                    layers += [nn.Conv2d(in_channels=in_channels,out_channels=v[1],kernel_size=(v[0]))]
                    # layers += [nn.BatchNorm2d(num_features=v[1])]

                layers += [nn.LeakyReLU(0.1)]
                in_channels = v[1]

            elif type(v) == list:
                conv1 = v[0]  # Tuple
                conv2 = v[1]  # Tuple
                repeats = v[2]  # Int

                for _ in range(repeats):
                    layers += [nn.Conv2d(in_channels,conv1[1],kernel_size=conv1[0],stride=conv1[2],padding=conv1[3])]
                    layers += [nn.LeakyReLU(0.1)]
                    layers += [nn.Conv2d(conv1[1],conv2[1],kernel_size=conv2[0],stride=conv2[2],padding=conv2[3])]
                    layers += [nn.LeakyReLU(0.1)]
                    in_channels = conv2[1]

        self.model = nn.Sequential(*layers)
        self.fl0 = nn.Linear(in_features=1024 * 7 * 7,out_features=4096)
        self.leaky = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fl1 = nn.Linear(in_features=4096,out_features=30 * 7 * 7)

    def forward(self,x):
        x = self.model(x)
        x = x.view(-1)
        x = self.fl0(x)
        x = self.dropout(x)
        x = self.leaky(x)
        x = self.fl1(x)
        # In paper, no activation is used here
        x = x.view(30,7,7)
        return x

    def save(self,optimizer):
        """
        Responsible for saving the model's parameters
        :param optimizer: Optimizer used for the training
        """
        print("==> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint,self.config.filename)

    def load(self,optimizer,lr):
        """
        Responsible for loading the model's parameters
        :param optimizer: Raw optimizer waiting to be tuned
        :param lr: Learning rate
        :return: The optimizer with updated parameters
        """
        print("==> Loading checkpoint")
        checkpoint = torch.load(self.config.filename,map_location=self.config.device)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return optimizer
