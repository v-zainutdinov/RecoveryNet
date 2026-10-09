import torch
from torch import nn
from torch.utils import data
import torchvision.transforms.v2 as tfs
import torch.optim as optim

import os
from PIL import Image

from tqdm import tqdm

from models import UNetModel, DataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms=tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(mean=[0.5126603245735168, 0.42385348677635193, 0.3761342763900757],
                  std=[0.2756892442703247, 0.25184452533721924, 0.24568717181682587])
])

d_train=DataSet(transforms=transforms)
train_data=data.DataLoader(dataset=d_train, batch_size=8, shuffle=True, drop_last=False)

    
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
            
        