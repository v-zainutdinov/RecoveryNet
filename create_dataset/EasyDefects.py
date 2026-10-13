import numpy as np
import cv2

from DefectUtils import finish, pick, pick_int, density, luma, light_dark, Overlay, random_walk


# Параметры можно не передавать (None) — тогда они берутся случайно из диапазона в docstring.
# q: качество JPEG, 0 - без JPEG (нужно для цепочек эффектов)
class EasyDefects:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array

    def apply_film_grain_pipeline(self, sigma: float=None, grain_size: float=None, q: int=None) -> np.ndarray: # Легкое
        """Плёночное зерно: шум с размером зерна, сильнее в средних тонах
            Args:
                sigma: сила зерна (5-15)
                grain_size: размер зерна в пикселях (0.5-1.5)
                q: сжатие JPEG (55-90)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        sigma = pick(sigma, 5, 15)
        grain_size = pick(grain_size, 0.5, 1.5)

        # 1. Шум с размером зерна (а не попиксельный)
        noise = np.random.randn(h, w).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), grain_size)
        noise /= noise.std() + 1e-6

        # 2. Зерно сильнее в средних тонах, слабее в чистых тенях и светах
        y = luma(img) / 255.0
        weight = 0.3 + 0.7 * 4 * y * (1 - y)
        img = img + noise[..., None] * sigma * weight[..., None]

        # 3. JPEG
        return finish(img, q, 55, 90)

    def apply_dust_pipeline(self, n_dust: int=None, n_hairs: int=None, q: int=None) -> np.ndarray: # Легкое
        """Пыль (тёмные и светлые точки) и волоски (тонкие извилистые линии)
            Args:
                n_dust: количество пылинок (40-160 для 512x512, на меньших картинках меньше)
                n_hairs: количество волосков (0-3)
                q: сжатие JPEG (55-90)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        m = min(h, w)
        s = max(m / 512.0, 0.5)

        n_dust = max(1, int(pick_int(n_dust, 40, 160) * density(h, w)))
        n_hairs = pick_int(n_hairs, 0, 3)
        light, dark = light_dark(img)
        ov = Overlay(h, w)

        # 1. Пыль
        for _ in range(n_dust):
            c = (np.random.randint(0, w), np.random.randint(0, h))
            col = dark if np.random.rand() < 0.5 else light
            ov.circle(c, max(1, int(np.random.randint(1, 3) * s + 0.5)), col, np.random.uniform(0.5, 0.95))

        # 2. Волоски
        for _ in range(n_hairs):
            start = (np.random.uniform(0, w), np.random.uniform(0, h))
            pts = random_walk(start, np.random.uniform(0, 2 * np.pi), np.random.uniform(0.05, 0.25) * m, 12, 0.35)
            col = dark if np.random.rand() < 0.7 else light
            ov.polyline(pts, col, np.random.uniform(0.45, 0.8), 1)

        # 3. Накладываем слой (с мягкими краями) и JPEG
        img = ov.apply(img, blur=0.6)
        return finish(img, q, 55, 90)

    def apply_soft_scan_pipeline(self, blur_sigma: float=None, down_factor: float=None, q: int=None) -> np.ndarray: # Легкое
        """Мягкий скан: размытие + уменьшение и увеличение обратно
            Args:
                blur_sigma: сигма гауссова размытия (0.6-1.6)
                down_factor: во сколько раз временно уменьшаем (0.6-0.9)
                q: сжатие JPEG (40-80)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        blur_sigma = pick(blur_sigma, 0.6, 1.6)
        down_factor = pick(down_factor, 0.6, 0.9)

        # 1. Размытие (потеря резкости отпечатка)
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

        # 2. Downscale (INTER_AREA) и Upscale строго в исходный размер
        small = cv2.resize(img, (max(2, int(w * down_factor)), max(2, int(h * down_factor))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

        # 3. JPEG
        return finish(img, q, 40, 80)

    def apply_fade_pipeline(self, contrast: float=None, shift: float=None, q: int=None) -> np.ndarray: # Легкое
        """Выцветание: низкий контраст и сдвиг яркости (цвет/тон не меняется)
            Args:
                contrast: множитель контраста вокруг среднего (0.5-0.85)
                shift: сдвиг яркости (-15..25)
                q: сжатие JPEG (55-90)
        """
        img = self.image_array.astype(np.float32)

        contrast = pick(contrast, 0.5, 0.85)
        shift = pick(shift, -15, 25)

        # 1. Контраст вокруг среднего (на каждый канал одинаково, поэтому тон сохраняется)
        mean = img.mean(axis=(0, 1))
        img = (img - mean) * contrast + mean + shift

        # 2. JPEG
        return finish(img, q, 55, 90)
