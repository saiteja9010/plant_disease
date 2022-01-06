import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image
from torchvision import models
import torch.optim as optim
import copy
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes

# import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary              # for getting the summary of our model


import torchvision

import torch.nn as nn

# load model

def mode(model,output,pretrain=False):
    model=model
    pretrain=pretrain
    output=output
    if pretrain:
        model=model(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    # for param in model.layer1.parameters():
    #   param.requires_grad = True
        final_in_features = model.fc.in_features
        model.fc = nn.Linear(final_in_features,output)
        for param in model.parameters():
            if param.requires_grad:
                print(param.shape)
    else:
        model=model(num_classes=output)
    # print(model)
    # print(type(model))
    # model=model.to(device)
    return model
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.input_size = input_size
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, num_classes)  
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         # no activation and no softmax at the end
#         return out

# input_size = 784 # 28x28
# hidden_size = 500 
num_classes = 38
model = models.resnet34(num_classes=num_classes)

PATH = "sai1"
model.load_state_dict(torch.load(PATH))
model.eval()
# model.fc.out_features=num_classes

# model=mode(models.resnet18,38,pretrain=True)

# PATH = "F:\\resnet-flask-webapp-main\\resnet-flask-webapp-main\\sai1"
# model.load_state_dict(torch.load(PATH))
# model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
                                    # transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((225,225)),
                                    transforms.CenterCrop(200),
									# transforms.ToTensor(),
									# transforms.RandomInvert(0.4),
									# transforms.RandomHorizontalFlip(0.5),
									# transforms.RandomVerticalFlip(0.4),
									transforms.ColorJitter(brightness=[0.4,0.7]),
									transforms.Pad(padding=2,padding_mode="reflect"),
                                    transforms.ToTensor()
                                    # transforms.Normalize((0.1307,),(0.3081,))
                                    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    # images = image_tensor.reshape(-1, 28*28)
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted[0]
