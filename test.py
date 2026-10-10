import torch
from torch.utils import data
import torchvision.transforms.v2 as tfs
from PIL import Image
import os

from models import DataSet, UNetModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(mean=[0.5126603245735168, 0.42385348677635193, 0.3761342763900757],
                  std=[0.2756892442703247, 0.25184452533721924, 0.24568717181682587])
])

def denormalize(img_tensor):
    mean = torch.tensor([0.5126603245735168, 0.42385348677635193, 0.3761342763900757]).view(3, 1, 1)
    std = torch.tensor([0.2756892442703247, 0.25184452533721924, 0.24568717181682587]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)

# Загружаем тренировочный датасет
d_train = DataSet(transforms=transforms, train=True)

# Загружаем модель (важно: map_location='cpu')
model = UNetModel()
st = torch.load("model_unet_67_0012.tar", weights_only=True, map_location='cpu')
model.load_state_dict(st)
model.to(device)
model.eval()

# Создаём отдельную папку для результатов
output_dir = "all_predictions"
os.makedirs(output_dir, exist_ok=True)

# Прогоняем все 10 картинок
with torch.no_grad():
    for idx in range(len(d_train)):
        x, y = d_train[idx]
        x_batch = x.unsqueeze(0).to(device)
        
        predict = model(x_batch)[0].cpu()
        
        pred_img = denormalize(predict)
        orig_img = denormalize(y)
        dist_img = denormalize(x)
        
        # Одну склеенную картинку
        w = pred_img.shape[2]
        h = pred_img.shape[1]
        combined = torch.zeros(3, h, w * 3)
        combined[:, :, 0:w]     = dist_img
        combined[:, :, w:2*w]   = pred_img
        combined[:, :, 2*w:3*w] = orig_img
        tfs.ToPILImage()(combined).save(os.path.join(output_dir, f"img_{idx:02d}_combined.png"))
        
        print(f"Сохранено: img_{idx:02d}")

print(f"\nГотово! Все результаты в папке: {output_dir}/")
print(f"Всего файлов: {len(os.listdir(output_dir))}")