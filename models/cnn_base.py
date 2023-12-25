import torch.nn as nn
import torch

from torchstat import stat

"""
Tropical Cyclone Intensity Estimation From
Geostationary Satellite Imagery Using Deep
Convolutional Neural Networks
"""

class CNNBase(nn.Module):

  def __init__(self):
    super(CNNBase, self).__init__()
    self.conv1 = nn.Conv2d(4, 16, (10, 10), stride=3, padding=0)
    self.pool1 = nn.MaxPool2d((3, 3), stride=2, padding=0)
    self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
    self.pool2 = nn.MaxPool2d(3, 2, 0)
    self.conv3 = nn.Conv2d(32, 64, 3, 1, 2)
    self.pool3 = nn.MaxPool2d(3, 2, 0)
    self.conv4 = nn.Conv2d(64, 128, 3, 1, 2)
    self.pool4 = nn.MaxPool2d(3, 2, 0)
    self.dropout = nn.Dropout()
    self.fc1 = nn.Linear(3200, 1024)
    self.fc2 = nn.Linear(1024, 8)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    batch_size = x.shape[0]
    x1 = self.conv1(x)
    x1 = self.relu(x1)
    x1 = self.pool1(x1)

    x2 = self.conv2(x1)
    x2 = self.relu(x2)
    x2 = self.pool2(x2)

    x3 = self.conv3(x2)
    x3 = self.relu(x3)
    x3 = self.pool3(x3)

    x4 = self.conv4(x3)
    x4 = self.relu(x4)
    x4 = self.pool4(x4)

    x4 = x4.view(batch_size, -1)
    x5 = self.dropout(x4)
    x5 = self.fc1(x5)
    x5 = self.relu(x5)

    x6 = self.dropout(x5)
    x6 = self.fc2(x6)
    x6 = self.relu(x6)

    x7 = self.dropout(x6)
    out = self.softmax(x7)

    return out
  
if __name__ == "__main__":
  input = torch.randn(32, 4, 259, 259)
  model = CNNBase()
  output = model(input, 32)
  print(output.size())
  # stat(model, (4, 259, 259))