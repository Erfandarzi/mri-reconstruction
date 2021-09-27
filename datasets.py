#
import glob
import random
import os
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',size=(287,287)):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.size=size
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        # item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        with open(self.files_A[index % len(self.files_A)], "rb") as f_in:
              item_A=pickle.load(f_in)
              item_A_resized=np.zeros((30,self.size,self.size))
          
              for i in range (len(item_A)):
                  item_A_pil= Image.fromarray(item_A[i])
                  item_A_resized[i]= np.array(item_A_pil.resize((self.size,self.size),Image.BICUBIC))

        item_A= torch.tensor(item_A_resized)
      
        if self.unaligned:
              with open(self.files_B[random.randint(0, len(self.files_B) - 1)], "rb") as f_in:
                 item_B=pickle.load(f_in)
                 item_B_resized=np.zeros((30,self.size,self.size))
          
                 for i in range (len(item_B)):
                      item_B_pil= Image.fromarray(item_B[i])
                      item_B_resized[i]= np.array(item_B_pil.resize((self.size,self.size),Image.BICUBIC))
              item_B= torch.tensor(item_B_resized)
               #  item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
              with open(self.files_B[index % len(self.files_B)], "rb") as f_in:

                 item_B=pickle.load(f_in)
                 item_B_resized=np.zeros((30,self.size,self.size))
          
                 for i in range (len(item_B)):
                      item_B_pil= Image.fromarray(item_B[i])
                      item_B_resized[i]= np.array(item_B_pil.resize((self.size,self.size),Image.BICUBIC))
              item_B= torch.tensor(item_B_resized)

                

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))