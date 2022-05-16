from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision.transforms as transforms
from PIL import Image
import csv
import random

from matplotlib import pyplot as plt

train_file = '~/models/FCN/dataset.csv'
means = np.array([103.939, 116.779, 123.68]) / 255
h, w      = 1080, 1440
train_h   = 1056  
train_w   = 1440  
val_h     = 1056  
val_w     = w 

class DeepSeaDataset(Dataset):
    def __init__(self, image_paths, phase, n_class=2, crop = True):
        
        self.image_paths = image_paths
        self.n_class = n_class
        self.transforms = transforms.ToTensor()
        self.means = means
        self.crop = crop

        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        elif phase == 'val':
            self.new_h = val_h
            self.new_w = val_w

    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = np.asarray(Image.open(self.image_paths[index][0]).convert('RGB'))
        mask =  np.asarray(Image.open(self.image_paths[index][1]))/255 
        if self.crop:
            h, w, _ = image.shape
            top   = 23
            left  = 0
            image   = image[top:top + self.new_h, left:left + self.new_w]
            mask = mask[top:top + self.new_h, left:left + self.new_w]

        image = image[:, :, ::-1] #convert to BGR
        image = np.transpose(image, (2, 0, 1)) / 255.
        image[0] -= self.means[0]
        image[1] -= self.means[1]
        image[2] -= self.means[2]

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        
        h = mask.size()[0]
        w = mask.size()[1]
        
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][mask == c] = 1
        
        sample = {'X': image, 'Y': target, 'm': mask}
      
        return sample


if __name__ == "__main__":
    
    with open(train_file, newline = '') as f:
        reader = csv.reader(f)
        images = list(reader)
    train_data = DeepSeaDataset(image_paths=images, phase='train')
    
    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())
    
        # observe 4th batch
        if i == 3:
            plt.figure()
            plt.imshow(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
