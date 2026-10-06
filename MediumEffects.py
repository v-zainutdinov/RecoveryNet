import numpy as np
import cv2

class Medium:
    def __init__(self, image_array: np.ndarray):
            self.image_array=image_array
    
    def apply_motion_pipeline(self, k_size: int=15, noise_std: float=12, q: int=70) -> np.ndarray:
        """
        Motion blur + Гауссов шум + JPEG.
    
        Args:
            img: numpy массив BGR (H, W, 3), uint8
            k_size: длина "смаза" в пикселях (5-25)
            noise_std: сила Гауссова шума (5-25)
            q: качество JPEG (30-90)
        """
        img = self.image_array.astype(np.float32)
        
        k_size=np.random.randint(low=5, high=25)
        noise_std=np.random.uniform(low=5, high=25)
        q=np.random.randint(low=30, high=50)
    
        # 1. Motion Blur — рисуем линию сразу под нужным углом
        angle = np.random.uniform(0, 360)
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        center = k_size // 2
    
        rad = np.deg2rad(angle)
        dx = int(center * np.cos(rad))
        dy = int(center * np.sin(rad))
        pt1 = (center - dx, center - dy)
        pt2 = (center + dx, center + dy)
    
        cv2.line(kernel, pt1, pt2, color=1.0, thickness=1)
    
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        else:
            kernel[center, center] = 1.0  # fallback: нет размытия
    
        img = cv2.filter2D(img, -1, kernel)
    
        # 2. Noise (Гаусс)
        noise = np.random.normal(0, noise_std, img.shape)
        img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    
        # 3. JPEG (всегда последним)
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    def apply_lens_effects(self, vignette_scale: float=0.7, ca_shift: float=3, q: int=50) -> np.ndarray:
        """Солнечные зайчики
            Args:
                vignette_scale: ширина затемнения по краям (0.3-0.5)
                ca_shift: сдвиг красного и синего каналов (1-1.5)
                q: сжатие JPEG (60-90) 
        """
    
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        
        vignette_scale=np.random.uniform(low=0.3, high=0.5)
        ca_shift=np.random.uniform(low=1, high=1.5)
        q=np.random.randint(low=60,high=90)
        
        # 1. Виньетирование
        kernel_x = cv2.getGaussianKernel(w, w * vignette_scale)
        kernel_y = cv2.getGaussianKernel(h, h * vignette_scale)
        mask = kernel_y @ kernel_x.T
        mask = mask / mask.max()
        img = (img * mask[..., np.newaxis]).astype(np.uint8)
        
        # 2. Хроматическая аберрация (сдвиг каналов B и R)
        b, g, r = cv2.split(img)
        m_b = np.float32([[1, 0, ca_shift], [0, 1, 0]])
        m_r = np.float32([[1, 0, -ca_shift], [0, 1, 0]])
        b = cv2.warpAffine(b, m_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        r = cv2.warpAffine(r, m_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        img = cv2.merge([b, g, r])
    
        # 3. Lens flare (блик)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        flare = np.zeros((h, w), dtype=np.float32)
        cv2.circle(flare, (cx, cy), int(min(w, h) * 0.15), 1.0, -1)
        flare = cv2.GaussianBlur(flare, (0, 0), min(w, h) * 0.08)
        
        # Лучи блика
        for _ in range(3):
            p_len = np.random.randint(min(w, h) // 4, min(w, h) // 2)
            angle = np.random.uniform(0, 2 * np.pi)
            ex, ey = int(cx + p_len * np.cos(angle)), int(cy + p_len * np.sin(angle))
            cv2.line(flare, (cx, cy), (ex, ey), 0.5, thickness=np.random.randint(2, 6))
        
        flare = cv2.GaussianBlur(flare, (0, 0), 5)
        img = np.clip(img.astype(np.float32) + flare[..., np.newaxis] * 120, 0, 255).astype(np.uint8)
        
        # 4. JPEG
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    def apply_old_photo_pipeline(self, k: int=3, q: int=40) -> np.ndarray:
        """Эффект старого фото с царапинами и точками
            Args:
                k: размер матрицы gaussing blur
                q: сжатие jpeg
        """
        img = self.image_array.astype(np.float32)
        
        matrix = np.array([
            [0.4, 0.4, 0.1],  # Новый Синий (B)
            [0.3, 0.5, 0.3],  # Новый Зеленый (G)
            [0.2, 0.4, 0.6]   # Новый Красный (R)
        ])
        img = cv2.transform(img, matrix)

        img = cv2.GaussianBlur(img, (k, k), 0)

        h, w = img.shape[:2]
        # Царапины (случайные тонкие белые/серые линии)
        for _ in range(np.random.randint(2, 6)):
            p1 = (np.random.randint(0, w), np.random.randint(0, h))
            p2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = np.random.randint(180, 240)
            cv2.line(img, p1, p2, (color, color, color), thickness=1, lineType=cv2.LINE_AA)
        
        # Пыль (случайные черные и белые точки)
        num_dust = np.random.randint(50, 200)
        for _ in range(num_dust):
            pt = (np.random.randint(0, w), np.random.randint(0, h))
            color = 0 if np.random.rand() > 0.5 else 255
            cv2.circle(img, pt, radius=np.random.randint(1, 3), color=(color, color, color), thickness=-1)

        # JPEG
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    