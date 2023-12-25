import time
import torch.nn as nn
import torch

from torchstat import stat

class PHURIE(nn.Module):

  def __init__(self):
    super(PHURIE, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 5, 1, 0)
    self.pool1 = nn.MaxPool2d(5, 2, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
    self.pool2 = nn.MaxPool2d(3, 2)
    self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
    self.pool3 = nn.MaxPool2d(3, 2)
    self.conv4 = nn.Conv2d(64, 64, 3, 1, 0)
    self.pool4 = nn.MaxPool2d(3, 1)
    self.conv5 = nn.Conv2d(64, 128, 3, 1, 0)
    self.pool5 = nn.MaxPool2d(3, 2)
    self.conv6 = nn.Conv2d(128, 128, 3, 1, 0)
    self.pool6 = nn.MaxPool2d(3, 2)
    self.fc1 = nn.Linear(1152, 512)
    self.fc2 = nn.Linear(512, 8)
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
    x4 = self.pool4(x4)

    x5 = self.conv5(x4)
    x5 = self.relu(x5)
    x5 = self.pool5(x5)

    x6 = self.conv6(x5)
    x6 = self.relu(x6)
    x6 = self.pool6(x6)

    x6 = x6.view(x6.size(0), -1)
    x7 = self.fc1(x6)
    x7 = self.fc2(x7)

    out = self.softmax(x7)

    return out

if __name__ == "__main__":
  data = torch.randn(1, 3, 224, 224)
  model = PHURIE()
  # torch.cuda.synchronize()
  # start = time.time()
  out = model(data)
  print(out.size())
  # torch.cuda.synchronize()
  # end = time.time()
  # stat(model, (3, 224, 224))
  
  # time_elapsed = end - start
  # print('Training complete in {:.0f}m {:.0f}s'.format(
  #     time_elapsed // 60, time_elapsed % 60))