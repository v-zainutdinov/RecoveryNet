import numpy as np
import cv2

class Easy:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array
    
    def apply_resize_pipeline(self, down_factor=0.85, q=50) -> np.ndarray: # Легкое
        """Уменьшение и увеличивание изображения
            Args:
            down_factor: уменьшение на 15% (0.85)
            q: сжатие JPEG (50-90)
        """
        
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        
        q = np.random.randint(low=50, high=90)
        
        # 1. Downscale (Уменьшение через INTER_AREA)
        w_down, h_down = int(w * down_factor), int(h * down_factor)
        img = cv2.resize(img, (w_down, h_down), interpolation=cv2.INTER_AREA)
        
        # 2. Upscale (Возвращаем СТРОГО в исходный размер оригинала w, h)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 3. JPEG
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


    def apply_posterization(self, levels: int=32, q: int=60) -> np.ndarray: #Лёгкое
            """Лёгкая постеризация (едва заметные цветовые ступеньки).
                Args:
                    levels: уровень постеризации (12-18 от заметного)
                    q: сжатие JPEG (60-90)
            """
            img = self.image_array.astype(np.float32)
            
            levels=np.random.randint(low=12, high=18)
            q=np.random.randint(low=60, high=90)
            
            # 1. Квантование цвета (Постеризация)
            shift = 256 // levels
            img = (img // shift) * shift + (shift // 2)
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # 2. JPEG
            _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)