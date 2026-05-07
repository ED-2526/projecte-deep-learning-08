import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,128,3,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(128,256,3,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(256,512,3,padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2,1),(2,1))
        self.conv5 = nn.Conv2d(512,512,3,padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d((2,1),(2,1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x))); x = self.pool5(x)
        x = x.squeeze(2).permute(2,0,1)
        return x

class CRNN(nn.Module):
    def __init__(self, num_classes, img_height, img_width, hidden_size=256, dropout=0.2):
        super().__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(512, hidden_size, 2, bidirectional=True, batch_first=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        feat = self.cnn(x)
        out, _ = self.rnn(feat)
        out = self.dropout(out)
        out = self.fc(out)
        return out