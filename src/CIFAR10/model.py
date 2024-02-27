from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1)
        self.dropout = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.linear0 = nn.Linear(3 * 3 * 256, 256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x, targets):
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1, 3 * 3 * 256)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        logits = self.linear3(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss