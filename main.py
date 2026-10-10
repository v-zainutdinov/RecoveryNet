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

d_train=DataSet(transforms, "Train")
train_data=data.DataLoader(dataset=d_train, batch_size=8, shuffle=True, drop_last=False)

v_train=DataSet(transforms, "Val")
val_data=data.DataLoader(dataset=v_train, batch_size=8, shuffle=True, drop_last=False)

    
model = UNetModel()
model.to(device)

opt=optim.Adam(model.parameters(), lr=0.0001)
fn_loss=nn.L1Loss()

#Цикл обучения

EPOCH = 80
patience = 7
min_delta = 1e-3
best_val= float("inf")
bad_epochs = 0
best_path = os.path.join("models_save", "model_unet_best.pt")
os.makedirs("model_save", exist_ok=True)

mult=1.2
best_loss=1e10

for _e in range(EPOCH):
    model.train()
    train_tqdm = tqdm(train_data, leave=False)
    
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
                
    model.eval()
    Q_val=0
    count_val=0
    
    for x_val, y_val in val_data:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            predict=model(x_val)
            loss=fn_loss(predict, y_val)
            
            Q_val+=loss.item()
            count_val+=1
            
            if best_loss > loss_mean*mult:
                best_loss=loss_mean
                st=model.state_dict()
                torch.save(st, os.path.join("models_save",f'model_unet_{_e}_{best_loss:.3f}'))
            if _e == 20:
                mult=1.01
            
    
    Q_val /= count_val
    
    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")
    
    if best_val - Q_val > min_delta:
        best_val = Q_val
        bad_epochs = 0
        torch.save(model.state_dict(), best_path)
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Эпоха {_e}. Val loss не улучшается {patience} эпох, обучение остановлено.")
            break
        
model.load_state_dict(torch.load(best_path, map_location=device))        
model.eval()

t_train=DataSet(transforms, "Test")
test_data =data.DataLoader(dataset=t_train, batch_size=16, shuffle=False, drop_last=False)

Q=0

torch.cuda.empty_cache()
with torch.no_grad():
    for x_test, y_test in test_data:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        predict=model(x_test)
        predict=torch.argmax(predict, dim=1)
        Q += torch.sum(predict == y_test).item()
        
Q /= len(t_train)
print(Q)