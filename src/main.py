import torch
from torch import nn
from torch.utils import data
import torchvision.transforms.v2 as tfs
import torch.optim as optim

import os

from tqdm import tqdm

from src.models import UNetModel, DataSet, CustomLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms=tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(mean=[0.48770615458488464, 0.4506271779537201, 0.4115591049194336],
                  std=[0.2185600996017456, 0.21286065876483917, 0.21262872219085693])
])

d_train=DataSet(transforms, "Train")
train_data=data.DataLoader(dataset=d_train, batch_size=8, shuffle=True, drop_last=False)

v_train=DataSet(transforms, "Val")
val_data=data.DataLoader(dataset=v_train, batch_size=8, shuffle=True, drop_last=False)

    
model = UNetModel()
model.to(device)

opt=optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
fn_loss=CustomLoss().to(device)

#Цикл обучения

EPOCH = 80
patience = 7
min_delta = 1e-3
best_val= float("inf")
bad_epochs = 0
best_path = os.path.join("models_save", "model_unet_best.pt")
os.makedirs("models_save", exist_ok=True)

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
        
model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))        
model.eval()

t_train = DataSet(transforms, "Test")
test_data = data.DataLoader(dataset=t_train, batch_size=16, shuffle=False, drop_last=False)


total_loss = 0
count_batches = 0

torch.cuda.empty_cache()
with torch.no_grad():
    for x_test, y_test in test_data:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        predict = model(x_test)
        
        loss = fn_loss(predict, y_test)
        
        total_loss += loss.item()
        count_batches += 1
        
final_metric = total_loss / count_batches
print(final_metric)