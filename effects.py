import cv2
import os
import shutil
import numpy as np

from EasyEffects import Easy
from EasyAndMediumEffects import EasyAndMedium
from MediumEffects import Medium
from HardEffects import Hard

# 30% - чистые фото
# 15% - простые искажения
# 15% - просто-нормально искажения
# 25% - нормальные искажения
# 15% - сложные искажения

def get_boundaries(total: int, rations: list) -> list:
    boundaries = [0]
    for i, ration in enumerate(rations):
        if i == len(rations) - 1:
            boundaries.append(total)
        else:
            target = round(total * sum(rations[:i+1]) / 100)
            boundaries.append(target)
    return boundaries

os.makedirs("original_photos", exist_ok=True)
os.makedirs("distorted_photos", exist_ok=True)

original_path_faces=os.path.join("Train", "faces")

photos_original=os.listdir(original_path_faces)
print(f"Количество файлов: {len(photos_original)}")

count_photo=10

list_percent=[30, 15, 15, 25, 15]
bounds=get_boundaries(count_photo, list_percent)


for i in range(0,count_photo):
    # текущий файл из списка ['fol1.png', 'dw2.png' и тд]
    current_orignal_photo=photos_original[i]
    
    # берём 1.png, 2.png и тд
    new_name=f"{i+1}.png"
    
    shutil.copy(
        os.path.join(original_path_faces, current_orignal_photo), 
        os.path.join("original_photos", new_name)
    )
    
    #открытие файла
    open_image=cv2.imread(os.path.join("original_photos", new_name))
    
    #original_resize=cv2.resize(open_image, (256, 256), interpolation=cv2.INTER_AREA)
    #image256 = cv2.resize(original_resize, (256, 256), interpolation=cv2.INTER_AREA)
    
    if i < bounds[1]:  # 0 <= i < 3 (30%)
        cv2.imwrite(os.path.join("distorted_photos", new_name), open_image)
        
    elif bounds[1] <= i < bounds[2]:  # 3 <= i < 4 (15%)
        r = np.random.random()
        easy = Easy(open_image)
        if r < 0.5:
            cv2.imwrite(os.path.join("distorted_photos", new_name), easy.apply_resize_pipeline())
        else:
            cv2.imwrite(os.path.join("distorted_photos", new_name), easy.apply_posterization())
            
    elif bounds[2] <= i < bounds[3]:
        r = np.random.random()
        easyAndmedium = EasyAndMedium(open_image)
        if r < 0.5:
            cv2.imwrite(os.path.join("distorted_photos", new_name), easyAndmedium.apply_complex_effects())
        else:
            cv2.imwrite(os.path.join("distorted_photos", new_name), easyAndmedium.apply_wave_pipeline())
            
    elif bounds[3] <= i < bounds[4]:
        r = np.random.random()
        medium = Medium(open_image)
        if r < 0.3:
            cv2.imwrite(os.path.join("distorted_photos", new_name), medium.apply_motion_pipeline())
        elif 0.3 <= r < 0.7:
            cv2.imwrite(os.path.join("distorted_photos", new_name), medium.apply_old_photo_pipeline())
        else:
            cv2.imwrite(os.path.join("distorted_photos", new_name), medium.apply_lens_effects())
            
    else:
        hard = Hard(open_image)
        cv2.imwrite(os.path.join("distorted_photos", new_name), hard.apply_tone_mapping_failure())
        

#cv2.imwrite('1.jpg', es.apply_posterization())
#30/30/25/15



