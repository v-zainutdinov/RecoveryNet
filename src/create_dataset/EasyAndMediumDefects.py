import numpy as np
import cv2

from src.create_dataset.DefectUtils import (finish, pick, pick_int, density, light_dark, Overlay,
                         random_walk, smooth_field, coords)


# Параметры можно не передавать (None) — тогда они берутся случайно из диапазона в docstring.
# q: качество JPEG, 0 - без JPEG (нужно для цепочек эффектов)
class EasyAndMediumDefects:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array

    def apply_scratches_pipeline(self, n_scratches: int=None, alpha: float=None, q: int=None) -> np.ndarray: # Легкие <-> Средние
        """Царапины: длинные тонкие светлые и тёмные линии, местами прерывистые
            Args:
                n_scratches: количество царапин (3-10)
                alpha: непрозрачность царапины (0.3-0.8)
                q: сжатие JPEG (50-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        m = min(h, w)
        s = max(m / 512.0, 0.5)

        n_scratches = pick_int(n_scratches, 3, 10)
        light, dark = light_dark(img)
        ov = Overlay(h, w)

        for _ in range(n_scratches):
            # 70% царапин почти вертикальные (как от протяжки плёнки), остальные под любым углом
            if np.random.rand() < 0.7:
                angle = np.pi / 2 + np.random.normal(0, 0.06)
                length = np.random.uniform(0.3, 1.0) * h
                wobble = 0.02
                start = (np.random.uniform(0, w), np.random.uniform(0, h * 0.7))
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.uniform(0.3, 1.0) * m
                wobble = 0.05
                start = (np.random.uniform(0, w), np.random.uniform(0, h))

            pts = random_walk(start, angle, length, max(4, int(length / 12)), wobble)
            col = light if np.random.rand() < 0.65 else dark
            ov.polyline(pts, col, pick(alpha, 0.3, 0.8), 1 if s < 1.5 else 2, gap_prob=0.1)

        img = ov.apply(img, blur=0.6)
        return finish(img, q, 50, 85)

    def apply_salt_pepper_pipeline(self, p: float=None, k: int=None, q: int=None) -> np.ndarray: # Легкие <-> Средние
        """Соль-перец (светлые и тёмные точки-пиксели) + лёгкое размытие
            Args:
                p: доля испорченных пикселей (0.001-0.012)
                k: размер ядра GaussianBlur (3 или 5)
                q: сжатие JPEG (35-70)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        p = pick(p, 0.001, 0.012)
        k = int(np.random.choice([3, 5])) if k is None else int(k)
        light, dark = light_dark(img)

        # 1. GaussianBlur
        img = cv2.GaussianBlur(img, (k, k), 0)

        # 2. Salt-and-pepper: рисуем точки в слой (в сером тоне снимка), чтобы края были мягкие
        ov = Overlay(h, w)
        n = max(1, int(p * h * w))
        for _ in range(n):
            c = (np.random.randint(0, w), np.random.randint(0, h))
            col = dark if np.random.rand() < 0.5 else light   # Pepper / Salt
            ov.circle(c, np.random.randint(0, 2), col, np.random.uniform(0.6, 1.0))
        img = ov.apply(img, blur=0.5)

        # 3. JPEG
        return finish(img, q, 35, 70)

    def apply_paper_warp_pipeline(self, amp: float=None, cells: float=None, q: int=None) -> np.ndarray: # Легкие <-> Средние
        """Волнистость бумаги: гладкое случайное смещение пикселей (не синусоида)
            Args:
                amp: максимальное смещение в пикселях (2-8)
                cells: «крупность» волн, меньше - крупнее (3-6)
                q: сжатие JPEG (40-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        s = max(min(h, w) / 512.0, 0.5)

        amp = pick(amp, 2, 8) * s
        cells = pick(cells, 3, 6)

        # 1. Два гладких поля смещения по x и по y
        dx = (smooth_field(h, w, cells) - 0.5) * 2 * amp
        dy = (smooth_field(h, w, cells) - 0.5) * 2 * amp

        # 2. Remap
        x, y = coords(h, w)
        img = cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 3. JPEG
        return finish(img, q, 40, 85)

    def apply_scanner_streaks_pipeline(self, n_streaks: int=None, strength: float=None, q: int=None) -> np.ndarray: # Легкие <-> Средние
        """Полосы сканера: тонкие светлые и тёмные полосы вдоль всего кадра (вертикальные или горизонтальные)
            Args:
                n_streaks: количество полос (2-8)
                strength: яркость полосы в уровнях (6-25)
                q: сжатие JPEG (50-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        n_streaks = pick_int(n_streaks, 2, 8)
        strength = pick(strength, 6, 25)

        # 1. Горизонтальные полосы получаем транспонированием
        horizontal = np.random.rand() < 0.3
        if horizontal:
            img = img.transpose(1, 0, 2)
            h, w = w, h

        # 2. Полосы: ширина 1-4 px, по высоте интенсивность плывёт
        band = np.zeros((h, w), np.float32)
        for _ in range(n_streaks):
            x0 = np.random.randint(0, w)
            width = np.random.randint(1, 5)
            sign = 1 if np.random.rand() < 0.5 else -1
            mod = 0.4 + 0.6 * smooth_field(h, 2, 4)[:, 0]
            band[:, x0:x0 + width] += (sign * strength * mod)[:, None]
        band = cv2.GaussianBlur(band, (0, 0), 0.8)
        img = img + band[..., None]

        if horizontal:
            img = img.transpose(1, 0, 2)

        # 3. JPEG
        return finish(np.ascontiguousarray(img), q, 50, 85)
