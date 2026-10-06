import numpy as np
import cv2

class EasyAndMedium:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array
        
    def apply_complex_effects(self, k: int=5, p: float=0.005, gamma: float=0.7, q: int=50) -> np.ndarray:
        '''
        Args:
            k: размер ядра для GaussianBlur (по умолчанию 5)\n
            p: вероятность появления соли/перца (% пикселей) для 0.005=0.5% \n
            gamma: гамма-коррекция (1 < = осветление теней)\n
            q: качество JPEG сжатия (0-100)
        '''
        img = self.image_array.astype(np.float32)
        
        p=np.random.uniform(low=0.001, high=0.02)
        gamma=np.random.uniform(low=0.4, high=0.8)
        q=int(np.random.uniform(low=35, high=70))
        
        # 1. GaussianBlur
        img = cv2.GaussianBlur(img, (k, k), 0)
        
        # 2. Salt-and-pepper
        rnd = np.random.rand(*img.shape[:2])
        img[rnd < p / 2] = 128     # Pepper
        img[rnd > 1 - p / 2] = 200 # Salt
        
        # 3. Gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        # ВАЖНО: Конвертируем в uint8 перед LUT!
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, table)
        
        # 4. JPEG compression
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    def apply_wave_pipeline(self, amp: float=3, freq: float=0.03, q: int=70) -> np.ndarray:  # Легкие <-> Средние
        """Мягкий эффект ряби без сильного сглаживания
        Args:
                apm: длина волны (1.5-5.0) 
                freg: частота (0.01-0.03) 
                q: качество JPEG (30-90)
        """
        amp = np.random.uniform(1.5, 5.0)     
        freq = np.random.uniform(0.01, 0.03)  
        q = np.random.randint(30, 90)   
        
        img = self.image_array.astype(np.float32)
        
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        map_x = (x + amp * np.sin(y * freq)).astype(np.float32)
        map_y = (y + amp * np.cos(x * freq)).astype(np.float32)
        
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)