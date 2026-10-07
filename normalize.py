import torchvision.transforms.v2 as tfs
import torch
from torch.utils import data

from PIL import Image
import os

class DataSet(data.Dataset):
    def __init__(self, transforms):
        path_train="Train"
        path_distorted_photo="distorted_photos"
        path_original_photo="original_photos"
        self.transforms=transforms
        
        self.distorted_path=[]
        self.original_path=[]
        
        for filename in sorted(os.listdir(os.path.join(path_train, path_distorted_photo))):
            self.distorted_path.append(os.path.join(path_train, path_distorted_photo, filename))
            self.original_path.append(os.path.join(path_train, path_original_photo, filename))
    def __getitem__(self, item):
        d_path=self.distorted_path[item]
        o_path=self.original_path[item]
        
        d_img=Image.open(d_path)
        o_img=Image.open(o_path)
        
        if self.transforms:
            d_img=self.transforms(d_img)
            o_img=self.transforms(o_img)
        
        return d_img, o_img
    
    def __len__(self):
        return len(self.distorted_path)
            
transforms=tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])

d_train=DataSet(transforms=transforms)
train_data = data.DataLoader(dataset=d_train, batch_size=32, shuffle=True, drop_last=False)

channels_sum=0
channels_squared_sum=0
num_batches=0

for x,y in train_data:
    channels_sum+=torch.mean(x, dim=[0,2,3])
    channels_squared_sum+=torch.mean(x**2, dim=[0,2,3])
    
    num_batches+=1

mean=channels_sum/num_batches
variance=(channels_squared_sum/num_batches)-(mean**2)
std=torch.sqrt(variance)

print(f"mean (среднее) = {mean.tolist()}; std (отклонение) = {std.tolist()}")

# mean (среднее) = [0.5126603245735168, 0.42385348677635193, 0.3761342763900757]
# std (отклонение) = [0.2756892442703247, 0.25184452533721924, 0.24568717181682587]
