import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import h5py

class TCIR(Dataset):
  def __init__(self, dataset_dir: str, transform=None, multi_modal=False, dim=3) -> None:

    self.dataset_dir = Path(dataset_dir)
    self.multi_modal = multi_modal
    # load "info" as pandas dataframe
    self.data_info = pd.read_hdf(self.dataset_dir, key="info", mode='r')
    self.data_matrix = h5py.File(self.dataset_dir, 'r')['matrix']
    self.transform = transform
    self.dim = dim

  def __len__(self):
    return len(self.data_info)

  def AvoidDamagedVal(self, matrix):
    NanVal = np.where(matrix==np.NaN)
    LargeVal = np.where(matrix>1000)
    DemagedVal = [NanVal, LargeVal]
    for item in DemagedVal:
      for idx in range(len(item[0])):
        i = item[0][idx]
        j = item[1][idx]
        allValidPixel = []
        for u in range(-2,2):
          for v in range(-2,2):
            if (i+u) < 201 and (j+v) < 201 and not np.isnan(matrix[i+u,j+v]) and not matrix[i+u,j+v] > 1000:
              allValidPixel.append(matrix[i+u,j+v])
        if len(allValidPixel) != 0:
          matrix[i][j] = np.mean(allValidPixel)

    return matrix

  def __getitem__(self, index):        
    # id = self.data_info.iloc[index].loc['ID']
    vmax = self.data_info.iloc[index].loc['Vmax']
    # Lon = self.data_info.iloc[index].loc['lon']
    # Lat = self.data_info.iloc[index].loc['lat']
    # Time = self.data_info.iloc[index].loc['time']

    # Slice1: IR
    # Slice2: Water vapor
    # Slice3: VIS
    # Slice4: PMW

    ch_slice = self.data_matrix[index][:, :, 0]
    ch_slice = self.AvoidDamagedVal(ch_slice)

    img = np.zeros((201, 201, self.dim))

    if self.multi_modal:
      ch_slice1 = self.data_matrix[index][:, :, 1]
      ch_slice1 = self.AvoidDamagedVal(ch_slice1)
      # ch_slice2 = self.data_matrix[index][:, :, 2]
      #ch_slice2 = self.AvoidDamagedVal(ch_slice2)
      ch_slice3 = self.data_matrix[index][:, :, 3]
      ch_slice3 = self.AvoidDamagedVal(ch_slice3)

      # img[:, :, 0] = ch_slice # IR
      # img[:, :, 1] = ch_slice1# Water vapor
      # img[:, :, 2] = ch_slice3# PMW
      img[:, :, 0] = ch_slice # IR
      img[:, :, 1] = ch_slice1# Water vapor
      img[:, :, 2] = ch_slice3# PMW

      i = 3
      while i < self.dim:
        img[:, :, i] = ch_slice
        i += 1

    else: 
      img[:, :, 0] = ch_slice

      i = 1
      while i < self.dim:
        img[:, :, i] = ch_slice
        i += 1

    img = img.astype(np.uint8)
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    vmax = int(vmax)

    return img, vmax# , Lon, Lat, Time, id