"""
Fem CNN + RNN bidireccional

"""

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d((2, 1), (2, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = x.squeeze(2)          # (batch, features, time)
        x = x.permute(2, 0, 1)    # (time, batch, features)
        return x

class CRNN(nn.Module):
    def __init__(self, num_classes, img_height, img_width, hidden_size=256):
        super().__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size,
                           num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        out, _ = self.rnn(features)
        out = self.fc(out)
        return out