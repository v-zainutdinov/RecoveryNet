import torch
from torch import nn
from torch.utils import data
import torchvision.transforms.v2 as tfs
import torch.optim as optim

import os
from PIL import Image

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSet(data.Dataset):
    def __init__(self, transforms, train=True):
        path_train= "Train" if train else "Test"
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

transforms=tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(mean=[0.5126603245735168, 0.42385348677635193, 0.3761342763900757],
                  std=[0.2756892442703247, 0.25184452533721924, 0.24568717181682587])
])

d_train=DataSet(transforms=transforms)
train_data=data.DataLoader(dataset=d_train, batch_size=8, shuffle=True, drop_last=False)

#Архитектура модели
class UNetModel(nn.Module):
    class TwoLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        def forward(self, x):
            return self.block(x)
            
    class EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block=UNetModel.TwoLayers(in_channels, out_channels)
            self.max_pooling=nn.MaxPool2d(2)
        
        def forward(self, x):
            skip_c=self.block(x)
            down=self.max_pooling(skip_c)
            return down,skip_c
        
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose=nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, bias=True)
            self.block=UNetModel.TwoLayers(in_channels, out_channels)
            
        def forward(self, down, skip_c):
            down=self.transpose(down)
            up=torch.cat([down,skip_c], dim=1)
            return self.block(up)
    
    def __init__(self):
        super().__init__()
        self.enc_block1=self.EncoderBlock(3, 64)
        self.enc_block2=self.EncoderBlock(64, 128)
        self.enc_block3=self.EncoderBlock(128, 256)
        self.enc_block4=self.EncoderBlock(256, 512)
        
        self.bottleneck=self.TwoLayers(512, 1024)
        
        self.dec_block1=self.DecoderBlock(1024, 512)
        self.dec_block2=self.DecoderBlock(512, 256)
        self.dec_block3=self.DecoderBlock(256, 128)
        self.dec_block4=self.DecoderBlock(128, 64)
        
        self.out=nn.Conv2d(64, 3, 1, stride=1, padding=0)
        
        #self.sigmoid=nn.Sigmoid()
        
    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)
        
        x=self.bottleneck(x)
        
        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)
        
        return self.out(x)
    
model = UNetModel()
model.to(device)

opt=optim.Adam(model.parameters(), lr=0.0001)
fn_loss=nn.MSELoss()

#Цикл обучения

model.train()
EPOCH = 80

mult=1.2
best_loss=1e10

for _e in range(EPOCH):
    train_tqdm = tqdm(train_data, leave=True)
    
    loss_mean=0
    lm_count=0
    
    for x_train, y_train in train_tqdm:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        predict=model(x_train)
        loss=fn_loss(predict, y_train)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        lm_count+=1
        loss_mean=1/lm_count * loss.item() + (1 - 1/lm_count)*loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{EPOCH}], loss_mean={loss_mean:.3f}")
        
        if best_loss > loss_mean*mult:
            best_loss=loss_mean
            st=model.state_dict()
            torch.save(st, os.path.join("models_save",f'model_unet_{_e}_{best_loss:.3f}'))
        if _e == 20:
            mult=1.01
            
        