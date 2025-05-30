
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(128*14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.avg(x)
        # print(f"in fc1 pre: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"in fc1 post: {x.shape}")
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
