import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.deepti import DeepTI
from dataset.tcir import TCIR


def windspeed2classes(windspeed):
    # Transform into 7 classes
    # -1 = Tropical depression (W<34)
    # 0 = Tropical storm [34<W<64]
    # 1 = Category 1 [64<=W<83]
    # 2 = Category 2 [83<=W<96]
    # 3 = Category 3 [96<=W<113]
    # 4 = Category 4 [113<=W<137]
    # 5 = Category 5 [W >= 137]

    if windspeed <= 33:
       ws_class = 1
       return ws_class
    if windspeed <= 63:
       ws_class = 2
       return ws_class
    if windspeed <= 82:
       ws_class = 3
       return ws_class
    if windspeed <= 96:
       ws_class = 4
       return ws_class
    if windspeed <= 112:
       ws_class = 5
       return ws_class
    if windspeed <= 136:
       ws_class = 6
       return ws_class
    else:
       ws_class = 7
       return ws_class


class Warpper(Dataset):

  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
     return len(self.dataset)

  def __getitem__(self, item):
     img, label = self.dataset[item]
     label = windspeed2classes(label)

     return img, label

def make_dataset(args, train_trans=None, test_trans=None, class_flag=False):
  if train_trans is None:
    train_trans = transforms.Compose([
      transforms.Resize(args.reshape),
      transforms.RandomRotation(20),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.ToTensor(),
    ])

  if test_trans is None:
    test_trans = transforms.Compose([
      transforms.Resize(args.reshape),
      transforms.ToTensor(),
    ])

  if args.dataset == "TCIR":
    train_set = TCIR(os.path.join(args.dataset_root, "TCIR-train.h5"), transform=train_trans, dim=args.img_dim)
    if args.use_test:
      test_set = TCIR(os.path.join(args.dataset_root, "TCIR-test.h5"), transform=test_trans, dim=args.img_dim)
    else:
      test_set = TCIR(os.path.join(args.dataset_root, "TCIR-val.h5"), transform=test_trans, dim=args.img_dim)
  else:
    train_set = DeepTI(args.dataset_root, mode="Train", transform=train_trans, dim=args.img_dim)
    if args.use_test:
      test_set = DeepTI(args.dataset_root, mode="Test", transform=test_trans, dim=args.img_dim)
    else:
      test_set = DeepTI(args.dataset_root, mode="Val", transform=test_trans, dim=args.img_dim)

  if class_flag:
     train_set = Warpper(train_set)
     test_set = Warpper(test_set)

  train_loader = DataLoader(
    train_set, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
  )

  test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
  )

  return train_loader, test_loader


if __name__ == "__main__":
  # dset = TCIR(r"I:\TCIR-SPLT\TCIR-test.h5")
  dset = DeepTI(r"I:\NASA-tropical-storm", mode="Val")
  print(len(dset))
