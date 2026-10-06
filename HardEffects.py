import numpy as np
import cv2

class Hard:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array
        
    def apply_tone_mapping_failure(self) -> np.ndarray:
        """Провал HDR tone mapping: выжженные света, шумные тени, ореолы."""
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
    
        # 1. Сжимаем dynamic range — света выжжены, тени вытянуты
        shadows_lift = np.random.uniform(0.15, 0.35)   # насколько поднимаем тени
        highlights_crush = np.random.uniform(0.7, 0.9) # порог, после которого всё серое
    
        # Кривая tone mapping (S-образная, но кривая)
        img_norm = img / 255.0
        # Поднимаем тени
        img_norm = img_norm + shadows_lift * (1.0 - img_norm)
        # Сжимаем света
        mask = np.clip((img_norm - highlights_crush) / (1.0 - highlights_crush), 0, 1)
        img_norm = img_norm - mask * (img_norm - highlights_crush) * 0.8
        img = img_norm * 255.0
    
        # 2. Ореолы вокруг контрастных границ (halo)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), np.random.uniform(15, 40))
        edges = cv2.subtract(gray, blur)
        # Инвертированные ореолы — тёмные вокруг светлого, светлые вокруг тёмного
        halo = np.repeat(edges[:, :, np.newaxis], 3, axis=2).astype(np.float32)
        halo_strength = np.random.uniform(0.3, 0.7)
        img = img - halo * halo_strength
    
        # 3. Шум в тенях (усиление теней = усиление шума)
        brightness = np.mean(img, axis=2) / 255.0
        shadow_mask = 1.0 - np.clip(brightness * 1.5, 0, 1)
        noise_sigma = np.random.uniform(8.0, 20.0)
        noise = np.random.normal(0, noise_sigma, (h, w, 3)) * shadow_mask[:, :, np.newaxis]
        img += noise
    
        # 4. Лёгкая десатурация в светах (небо становится серым)
        highlight_mask = np.clip((brightness - 0.7) / 0.3, 0, 1)
        gray_3ch = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        img = img * (1.0 - highlight_mask[:, :, np.newaxis] * 0.5) + gray_3ch * highlight_mask[:, :, np.newaxis] * 0.5
    
        return np.clip(img, 0, 255).astype(np.uint8)