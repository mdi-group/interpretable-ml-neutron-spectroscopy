import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5, padding=0)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(16)

        
        self.conv3 = nn.Conv2d(16, 16, (1, 3), padding=(0, 0))
        self.conv4 = nn.Conv2d(16, 32, (1, 3), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(1445184, 64)

    def compute_features(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (1, 2), (1, 2))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x
    
class CNN_DUQ(ConvNet):
    def __init__(
        self,
        input_size,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 64), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale
            
    def update_embeddings(self, x, y):
        z = self.comp_last_layer(self.compute_features(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def comp_last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (-(diff ** 2)).mean(1).div(2 * self.sigma ** 2).exp()

        return distances

    def forward(self, x):
        z = self.comp_last_layer(self.compute_features(x))
        y_pred = self.output_layer(z)

        return z, y_pred
    
class SoftmaxModel(ConvNet):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.last_layer = nn.Linear(32, num_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred = F.log_softmax(z, dim=1)

        return y_pred


