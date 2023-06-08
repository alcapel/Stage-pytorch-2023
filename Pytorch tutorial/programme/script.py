from loop import train, test
from models import Net
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch import nn
import torchvision.transforms as transforms 

net = Net()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epochs = 10


trainset = CIFAR10(root='./data', train=True, download=True,transform=transforms.ToTensor())
trainCIFAR10 = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = CIFAR10(root='./data', train= False, download=True,transform=transforms.ToTensor())
testCIFAR10 = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)



for i in range(epochs):
  print(f"Epoch n°{i+1} ------------------------------\n")
  train(trainCIFAR10, net, optimizer, loss)
  test(testCIFAR10, net)

print("It's finished")