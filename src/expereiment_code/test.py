import torch
from torch.utils import data
import torchvision.transforms.v2 as tfs
from PIL import Image
import os

from src.models import DataSet, UNetModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(mean=[0.48770615458488464, 0.4506271779537201, 0.4115591049194336],
                      std=[0.2185600996017456, 0.21286065876483917, 0.21262872219085693])
])

def denormalize(img_tensor):
    mean = torch.tensor([0.48770615458488464, 0.4506271779537201, 0.4115591049194336]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2185600996017456, 0.21286065876483917, 0.21262872219085693]).view(3, 1, 1).to(device)
    return (img_tensor * std + mean).clamp(0, 1)


d_train = DataSet(transforms=transforms, path_train="Test")
test_dataset = data.DataLoader(dataset=d_train, batch_size=8, shuffle=True, drop_last=False)

# Загружаем модель (важно: map_location='cpu')
model = UNetModel()
st = torch.load(os.path.join("models_save", "model_unet_best.pt"), weights_only=True)
model.load_state_dict(st)
model.to(device)
model.eval()

# Создаём отдельную папку для результатов
output_dir = "all_predictions"
os.makedirs(output_dir, exist_ok=True)

current_photo = 0
limit_count_photo = 200

with torch.no_grad():
    for x_test, y_test in test_dataset:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        if current_photo == limit_count_photo:
            break
            
        predict = model(x_test)
                    
        pred_img = denormalize(predict)
        orig_img = denormalize(y_test)
        dist_img = denormalize(x_test)
                    
        # Проходим по каждому изображению в батче
        batch_size = x_test.shape[0]
        for i in range(batch_size):
            if current_photo >= limit_count_photo:
                break
                
            w = pred_img.shape[3]  # width
            h = pred_img.shape[2]  # height
            
            # Берём одно изображение из батча: [3, 512, 512]
            dist_single = dist_img[i]
            pred_single = pred_img[i]
            orig_single = orig_img[i]
            
            # Создаём комбинированное изображение
            combined = torch.zeros(3, h, w * 3)
            combined[:, :, 0:w]     = dist_single
            combined[:, :, w:2*w]   = pred_single
            combined[:, :, 2*w:3*w] = orig_single
            
            tfs.ToPILImage()(combined).save(
                os.path.join(output_dir, f"img_{current_photo:02d}_combined.png")
            )
            
            current_photo += 1

print(f"\nГотово! Все результаты в папке: {output_dir}/")
print(f"Всего файлов: {len(os.listdir(output_dir))}")