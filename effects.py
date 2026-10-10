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

#cv2.imwrite('1.jpg', es.apply_posterization())
#30/30/25/15

class CreateDataset:
    def __init__(self, count_img_train: int, count_img_val: int, count_img_test):
        self.count_img_train = count_img_train
        self.count_img_val = count_img_val
        self.count_img_test = count_img_test
        
    def get_boundaries(self, total: int, rations: list) -> list:
        boundaries = [0]
        for i, ration in enumerate(rations):
            if i == len(rations) - 1:
                boundaries.append(total)
            else:
                target = round(total * sum(rations[:i+1]) / 100)
                boundaries.append(target)
        return boundaries
    
    def create_folder(self):
        folder = ["Train", "Val", "Test"]
        
        for item in folder:
            os.makedirs(os.path.join(item, "original_photos"), exist_ok=True)
            os.makedirs(os.path.join(item,"distorted_photos"), exist_ok=True)
    
    def create_photo(self, original_photos: list, orginal_path: str, new_path_original: str, new_path_distorted: str) -> None:
        # Процентное соотношение
        train_list_percent = [30, 15, 15, 25, 15]
        
        # Получение индексов для разных типов дефектов
        bounds = self.get_boundaries(len(original_photos), train_list_percent)
                
        for i in range(len(original_photos)):
            # текущий файл из списка ['fol1.png', 'dw2.png' и тд]
            current_orignal_photo=original_photos[i]
            
            # берём 1.png, 2.png и тд
            new_name=f"{i+1}.png"
            
            shutil.copy(
                os.path.join(orginal_path, current_orignal_photo), 
                os.path.join(new_path_original, new_name)
            )
            
            #открытие файла
            open_image=cv2.imread(os.path.join(new_path_original, new_name))
            
            if i < bounds[1]:
                cv2.imwrite(os.path.join(new_path_distorted, new_name), open_image)
                
            elif bounds[1] <= i < bounds[2]:
                r = np.random.random()
                easy = Easy(open_image)
                if r < 0.5:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), easy.apply_resize_pipeline())
                else:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), easy.apply_posterization())
                    
            elif bounds[2] <= i < bounds[3]:
                r = np.random.random()
                easyAndmedium = EasyAndMedium(open_image)
                if r < 0.5:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), easyAndmedium.apply_complex_effects())
                else:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), easyAndmedium.apply_wave_pipeline())
                    
            elif bounds[3] <= i < bounds[4]:
                r = np.random.random()
                medium = Medium(open_image)
                if r < 0.3:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), medium.apply_motion_pipeline())
                elif 0.3 <= r < 0.7:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), medium.apply_old_photo_pipeline())
                else:
                    cv2.imwrite(os.path.join(new_path_distorted, new_name), medium.apply_lens_effects())
                    
            else:
                hard = Hard(open_image)
                cv2.imwrite(os.path.join(new_path_distorted, new_name), hard.apply_tone_mapping_failure())
    
    def create_dataset(self):
        self.create_folder()
        
        # Папка для обучающей выборке
        train_path_faces = os.path.join("Train", "faces")
        
        # Папка для выборке валидации и тестировании
        val_test_faces = os.path.join("Test", "faces")

        # Файлы в папках
        train_photos_original = os.listdir(train_path_faces)[:self.count_img_train]
        val_test_photos_original = os.listdir(val_test_faces)[:self.count_img_val+self.count_img_test]

        half_size = len(val_test_photos_original) // 2
        
        # Разделение тестовой выборки на val/test
        val_photos_original = val_test_photos_original[:half_size]
        test_photos_original = val_test_photos_original[half_size:]

        self.create_photo(train_photos_original, train_path_faces, os.path.join("Train", "original_photos"), os.path.join("Train", "distorted_photos"))
        self.create_photo(val_photos_original, val_test_faces, os.path.join("Val", "original_photos"), os.path.join("Val", "distorted_photos"))
        self.create_photo(test_photos_original, val_test_faces, os.path.join("Test", "original_photos"), os.path.join("Test", "distorted_photos"))
        
if __name__ == "__main__":
    dataset=CreateDataset(1000, 500, 500)
    dataset.create_dataset()