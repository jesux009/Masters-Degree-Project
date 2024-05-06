import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

def img2tensor(img, dtype: np.dtype = np.float32):
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img.astype(dtype, copy=False))
    return tensor

class ContrailsDataset(Dataset):

     def __init__(self, path, use='train', soft_labels=False, only_positives=True, repeat=1):
          if use == 'train' or use=='metrics':
               train = True
          else:
               train = False
          self.path = os.path.join(path, "train" if train else "validation", "images")
          if only_positives:
               positives_path = r'C:\Users\USER\Desktop\UNIVERSIDAD\MÁSTER AERONÁUTICA - UC3M\SEGUNDO\TRABAJO DE FIN DE MÁSTER\Development\Inspection'
               positives_file = np.load(os.path.join(positives_path,"positive_train.npy" if train else "positive_validation.npy"))
               positives_fnames = [filename.split("\\")[3] for filename in positives_file]
               if use == 'train' or use == 'cross-validate':
                    self.filenames = [filename.split(".")[0] for filename in os.listdir(self.path) if filename.split(".")[0] in positives_fnames]
               elif use == 'metrics':
                    self.filenames = random.sample([filename.split(".")[0] for filename in os.listdir(self.path) if filename.split(".")[0] in positives_fnames], 500)
          else:
               if use == 'train' or use == 'cross-validate':
                    self.filenames = [filename.split(".")[0] for filename in os.listdir(self.path)]
               elif use == 'metrics':
                    self.filenames = random.sample([filename.split(".")[0] for filename in os.listdir(self.path)],500)
          self.train = train
          self.nc = 3
          self.repeat = repeat
          self.soft_labels = soft_labels

     def __len__(self):
          return self.repeat * len(self.filenames)
     
     def __getitem__(self, index):
          index = index % len(self.filenames)
          try:
               image = np.array(Image.open(os.path.join(self.path, self.filenames[index] + '.png')))
               if self.soft_labels:
                    mask  = np.load(os.path.join(self.path.replace('images','soft_label'), self.filenames[index] + '.npy'))
               else:
                    mask  = np.load(os.path.join(self.path.replace('images','ground_truth'), self.filenames[index] + '.npy'))
               image_tensor, mask_tensor = img2tensor(image/255), img2tensor(mask)   # Sizes 3x256x256 and 1x256x256  
               return image_tensor, mask_tensor
          except Exception as e:
               print(f"\n Error loading file: {e} \n")
               return None, None
          # image = image.reshape(*image.shape[:2], -1)         # For later on if several frames are loaded, creating an array of (H,W,C*T)
          # image = image.view(self.nc, -1, *image.shape[1:])   # For later on if several frames are loaded, coming back to a tensor of (C,T,H,W)    
