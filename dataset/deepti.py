import numpy as np
import os.path
from PIL import Image
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import json
import pickle


class DeepTI(Dataset):
  """自定义数据集"""

  def __init__(self, root: str, mode: str="Train", transform=None, dim=3):
    # Read From TxTFile

    self.root = Path(root)
    self.transform = transform
    self.dim = dim

    self.images_path = []
    self.images_class = []

    self.TrainImageList = []
    self.TestImageList = []

    self.TrainImageLabelList = []
    self.TestImageLabelList = []

    all_data_path = os.path.join(self.root, "AllData.pkl")

    if os.path.isfile(all_data_path):
      with open(all_data_path, 'rb') as file2:
        self.TrainImageList = pickle.load(file2)
        self.TrainImageLabelList = pickle.load(file2)
        self.TestImageList = pickle.load(file2)
        self.TestImageLabelList = pickle.load(file2)
    else:
      train_source = 'nasa_tropical_storm_competition_train_source'
      train_labels = 'nasa_tropical_storm_competition_train_labels'
      jpg_paths = glob(str(self.root / train_source / '**' / '*.jpg'))

      for jpg_path in jpg_paths:
        self.TrainImageList.append(jpg_path)
        jpg_path = Path(jpg_path)

        # Get the IDs and file paths
        features_path = jpg_path.parent / 'features.json'
        image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
        storm_id = image_id.split('_')[0]
        labels_path = str(jpg_path.parent / 'labels.json').replace(train_source, train_labels)

        # Load the features data
        with open(features_path) as src:
          features_data = json.load(src)

        # Load the labels data
        with open(labels_path) as src:
          labels_data = json.load(src)

        self.TrainImageLabelList.append(int(labels_data['wind_speed']))

      self.images_path = self.TrainImageList
      self.images_class = self.TrainImageLabelList

      test_source = 'nasa_tropical_storm_competition_test_source'
      test_labels = 'nasa_tropical_storm_competition_test_labels'

      jpg_paths = glob(str(self.root / test_source / '**' / '*.jpg'))

      for jpg_path in jpg_paths:
        self.TestImageList.append(jpg_path)
        jpg_path = Path(jpg_path)
        # Get the IDs and file paths
        features_path = jpg_path.parent / 'features.json'
        image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])

        labels_path = str(jpg_path.parent / 'labels.json').replace(test_source, test_labels)

        with open(labels_path) as src:
          labels_data = json.load(src)

        self.TestImageLabelList.append(int(labels_data['wind_speed']))

      self.images_path = self.TestImageList
      self.images_class = self.TestImageLabelList

      with open(all_data_path, 'wb') as file:
        pickle.dump(self.TrainImageList, file)
        pickle.dump(self.TrainImageLabelList, file)
        pickle.dump(self.TestImageList, file)
        pickle.dump(self.TestImageLabelList, file)

    if mode == "Train":
      self.images_path = self.TrainImageList
      self.images_class = self.TrainImageLabelList

    if mode == "Val":
      self.images_path = self.TestImageList
      self.images_class = self.TestImageLabelList

    if mode == "Test":
      self.images_path = self.TestImageList
      self.images_class = self.TestImageLabelList


  def __len__(self):
      return len(self.images_path)
  

  def __getitem__(self, item):
      img = cv2.imread(self.images_path[item])
      
      # RGB为彩色图片，L为灰度图片
      if img.shape[2] != 3:
          raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
      
      while img.shape[2] < self.dim:
        img = np.append(img, img[:, :, 0, None], axis=2)
      
      img = img.astype(np.uint8)
      img = Image.fromarray(img)
      label = self.images_class[item]

      if self.transform is not None:
          img = self.transform(img)

      return img, label
