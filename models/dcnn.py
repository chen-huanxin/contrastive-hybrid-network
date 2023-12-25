import torch.nn as nn
import torch

from torchstat import stat

class DCNN(nn.Module):

  def __init__(self):
    super(DCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 10, 3, 0)
    self.pool1 = nn.MaxPool2d(3, 2, 0)
    self.conv2 = nn.Conv2d(64, 256, 5, 1, 0)
    self.pool2 = nn.MaxPool2d(3, 2, 0)
    self.conv3 = nn.Conv2d(256, 288, 3, 1, 1)
    self.pool3 = nn.MaxPool2d(2, 1, 0)
    self.conv4 = nn.Conv2d(288, 272, 3, 1, 1)
    self.conv5 = nn.Conv2d(272, 256, 3, 1, 0)
    self.pool5 = nn.MaxPool2d(3, 2, 0)

    self.fc6 = nn.Linear(9216, 3584)
    self.fc7 = nn.Linear(3584, 2048)
    self.fc8 = nn.Linear(2048, 8)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
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
    x5 = self.conv5(x4)
    x5 = self.relu(x5)
    x5 = self.pool5(x5)

    x5 = x5.view(x5.size(0), -1)
    x6 = self.fc6(x5)
    x7 = self.fc7(x6)
    x8 = self.fc8(x7)

    out = self.softmax(x8)

    return out

if __name__ == "__main__":
  # data = torch.randn(1, 3, 232, 232)
  # model = DCNN()
  # # out = model(data)
  # # print(out.size())
  # stat(model, (3, 232, 232))

  import numpy as np
  a = np.ones((224, 224, 3))
  b = a[:, :, 0, None]
  print(b.shape)
  a = np.append(a, b, axis=2)
  print(a.shape)
