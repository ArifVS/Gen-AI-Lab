#importing the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision #torchvision PyTorch ka ek library hai jo images datasets, image models, aur image transformations ke liye use hota hai
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#
transform = transforms.Compose([transforms.ToTensor, transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
#transforms.ToTensor
# Ye image ko PyTorch tensor me convert karta hai aur pixel values ko 0-255 se 0-1 range me normalize karta hai.
'''#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
👉 Har channel (Red, Green, Blue) ke liye image ko normalize karta hai:
Mean = 0.5
Std (standard deviation) = 0.5
Iska matlab: final pixel values 0 se -1 aur 1 ke beech ho jaate hain.'''
#Load CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data',train = True, download = True,transform=transform)
#trainloader=Training dataset ko chhote-chhote batches me load karne wala tool.
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)
