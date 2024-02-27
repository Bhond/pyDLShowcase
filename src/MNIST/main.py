import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load datasets
trainingData = datasets.FashionMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

testingData = datasets.FashionMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

# Select device
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)


# Create model
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Init layers
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),  # 28 * 28 images, 512 why not?
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),  # 512 why not?
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)  # 512 why not? 10 labels
        )

    def forward(self, x):
        # Take input tensor and flatten it
        x = self.flatten(x)
        # Feed it to the stack and return a tensor which is the output layer
        logits = self.linearReluStack(x)
        # Return the result
        return logits


# Create the model
model = NeuralNetwork()

# Send it to the device
model.to(device)

# Define hyperparameters
batchSize = 64
epochs = 5
lr = 0.1

# Create data loaders
trainingDataloader = DataLoader(trainingData, batchSize)
testingDataloader = DataLoader(testingData, batchSize)

# Select loss function
lossFn = nn.CrossEntropyLoss()

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# Define training loop
def trainingLoop(dataloader, model, lossFn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode, check tutorial for full description of what this line does
    model.train()
    # Start loop: retrieves the batch idx, the data, the label
    for batch, (X, y) in enumerate(dataloader):
        # Forward pass
        pred = model.forward(X)
        # Retrieve loss
        loss = lossFn(pred, y)
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Add print -> Copied
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batchSize + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Define testing loop
def testingLoop(dataloader, model, lossFn):
    size = len(dataloader.dataset)
    numBatches = len(dataloader)
    testLoss, correct = 0, 0
    # Set the model to evaluation mode, check tutorial for full description of what this line does
    model.eval()
    # Start loop: retrieves the data, the label
    with torch.no_grad():
        for X, y in dataloader:
            # Forward pass
            pred = model.forward(X)
            # Loss
            testLoss += lossFn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # Compute score and display it
            testLoss /= numBatches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {testLoss:>8f}")


# Combine the loops
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    trainingLoop(trainingDataloader, model, lossFn, optimizer)
    testingLoop(testingDataloader, model, lossFn)
