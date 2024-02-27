"""
Class responsible for holding the training and testing
"""
from config import Config

class Trainer:
    def __init__(self, config):
        print("Ctor")

    def train(self):
        print("Train method")


#
# # Define training loop
# def trainingLoop(dataloader, model, lossFn, optimizer):
#     size = len(dataloader.dataset)
#     # Set the model to training mode, check tutorial for full description of what this line does
#     model.train()
#     # Start loop: retrieves the batch idx, the data, the label
#     for batch, (X, y) in enumerate(dataloader):
#         # Forward pass
#         pred = model.forward(X)
#         # Retrieve loss
#         loss = lossFn(pred, y)
#         # Backward
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # Add print -> Copied
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * batchSize + len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# # Define testing loop
# def testingLoop(dataloader, model, lossFn):
#     size = len(dataloader.dataset)
#     numBatches = len(dataloader)
#     testLoss, correct = 0, 0
#     # Set the model to evaluation mode, check tutorial for full description of what this line does
#     model.eval()
#     # Start loop: retrieves the data, the label
#     with torch.no_grad():
#         for X, y in dataloader:
#             # Forward pass
#             pred = model.forward(X)
#             # Loss
#             testLoss += lossFn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#             # Compute score and display it
#             testLoss /= numBatches
#             correct /= size
#             print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {testLoss:>8f}")
#
#
# # Combine the loops
# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}\n-------------------------------")
#     trainingLoop(trainingDataloader, model, lossFn, optimizer)
#     testingLoop(testingDataloader, model, lossFn)