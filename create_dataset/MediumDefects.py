import numpy as np
import cv2

from DefectUtils import (finish, pick, pick_int, density, luma, coords, smoothstep,
                         smooth_field, blob_distance, tone_colors, light_dark, Overlay)


# Параметры можно не передавать (None) — тогда они берутся случайно из диапазона в docstring.
# q: качество JPEG, 0 - без JPEG (нужно для цепочек эффектов)
class MediumDefects:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array

    def apply_stains_pipeline(self, n_stains: int=None, size: float=None, strength: float=None, q: int=None) -> np.ndarray: # Средние
        """Пятна (вода, жир): рваный край, тёмный ободок, структура под пятном видна
            Args:
                n_stains: количество пятен (1-4)
                size: радиус пятна, доля короткой стороны (0.1-0.35)
                strength: сила затемнения (0.25-0.6)
                q: сжатие JPEG (50-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        m = min(h, w)

        for _ in range(pick_int(n_stains, 1, 4)):
            r = pick(size, 0.1, 0.35) * m
            c = (np.random.uniform(0, w), np.random.uniform(0, h))
            st = pick(strength, 0.25, 0.6)

            # 1. Пятно с рваным краем + тёмное кольцо по краю (как у высохшей капли)
            d = blob_distance(h, w, c, r, rough=0.8)
            inside = 1 - smoothstep(0.8, 1.0, d)
            rim = np.exp(-((d - 0.95) / 0.06) ** 2)

            # 2. Затемняем умножением (тон снимка не меняется)
            dark = np.clip(inside * st * 0.5 + rim * st * 0.7, 0, 0.9)
            img = img * (1 - dark[..., None])

        # 3. JPEG
        return finish(img, q, 50, 85)

    def apply_uneven_fading_pipeline(self, fade: float=None, cells: float=None, edge_dark: float=None, q: int=None) -> np.ndarray: # Средние
        """Неравномерное выцветание: в одних местах блекло, в других нет + затемнённые края
            Args:
                fade: сила выцветания в самых выгоревших местах (0.3-0.7)
                cells: «крупность» неоднородности, меньше - крупнее (2-4)
                edge_dark: затемнение краёв (0.15-0.4)
                q: сжатие JPEG (50-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        fade = pick(fade, 0.3, 0.7)
        cells = pick(cells, 2, 4)
        edge_dark = pick(edge_dark, 0.15, 0.4)
        paper, _ = tone_colors(img)

        # 1. Карта выцветания: тянем к блеклому плоскому тону (среднее + бумага)
        mask = (smooth_field(h, w, cells) * fade)[..., None]
        target = 0.6 * img.mean(axis=(0, 1)) + 0.4 * paper
        img = img * (1 - mask) + target * mask

        # 2. Края темнее центра
        xx, yy = coords(h, w)
        edge_dist = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        e = (1 - np.clip(edge_dist / (0.15 * min(h, w)), 0, 1)) ** 2
        img = img * (1 - (e * edge_dark)[..., None])

        # 3. JPEG
        return finish(img, q, 50, 85)

    def apply_paper_texture_pipeline(self, strength: float=None, fiber_len: int=None, q: int=None) -> np.ndarray: # Средние
        """Фактура бумаги: направленные волокна + мелкое зерно + тиснение (emboss)
            Args:
                strength: амплитуда текстуры в уровнях яркости (6-16)
                fiber_len: длина волокон в пикселях (5-15)
                q: сжатие JPEG (50-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        strength = pick(strength, 6, 16)
        k = max(5, pick_int(fiber_len, 5, 15) | 1)

        # 1. Ядро-«волокно» под случайным углом
        c = k // 2
        ang = np.random.uniform(0, np.pi)
        kernel = np.zeros((k, k), np.float32)
        cv2.line(kernel, (c - int(c * np.cos(ang)), c - int(c * np.sin(ang))),
                 (c + int(c * np.cos(ang)), c + int(c * np.sin(ang))), 1.0, 1)
        kernel /= max(kernel.sum(), 1e-6)

        # 2. Волокна + мелкое зерно
        fib = cv2.filter2D(np.random.randn(h, w).astype(np.float32), -1, kernel)
        fib /= fib.std() + 1e-6
        fine = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (0, 0), 0.7)
        fine /= fine.std() + 1e-6
        tex = 0.6 * fib + 0.4 * fine

        # 3. Тиснение
        emboss_k = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], np.float32)
        emb = cv2.filter2D(tex, -1, emboss_k)
        emb /= emb.std() + 1e-6
        tex = 0.5 * tex + 0.5 * emb

        img = img + tex[..., None] * strength

        # 4. JPEG
        return finish(img, q, 50, 85)

    def apply_motion_pipeline(self, k_size: int=None, noise_std: float=None, q: int=None) -> np.ndarray: # Средние
        """Motion blur + Гауссов шум
            Args:
                k_size: длина "смаза" в пикселях (5-21)
                noise_std: сила Гауссова шума (4-14)
                q: сжатие JPEG (30-70)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        k_size = pick_int(k_size, 5, 21)
        noise_std = pick(noise_std, 4, 14)

        # 1. Motion Blur — рисуем линию сразу под нужным углом
        angle = np.random.uniform(0, 360)
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        center = k_size // 2

        rad = np.deg2rad(angle)
        dx = int(round(center * np.cos(rad)))
        dy = int(round(center * np.sin(rad)))
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), color=1.0, thickness=1)
        kernel = kernel / kernel.sum()

        img = cv2.filter2D(img, -1, kernel)

        # 2. Noise (Гаусс), одинаковый на всех каналах, чтобы не появлялся цветной шум
        img = img + np.random.normal(0, noise_std, (h, w, 1)).astype(np.float32)

        # 3. JPEG
        return finish(img, q, 30, 70)

    def apply_glare_vignette_pipeline(self, vignette_scale: float=None, glare: float=None, q: int=None) -> np.ndarray: # Средние
        """Виньетирование + блик (отражение на глянцевой бумаге при съёмке/скане)
            Args:
                vignette_scale: ширина виньетки, меньше - темнее края (0.5-1.0)
                glare: сила блика (0.25-0.6)
                q: сжатие JPEG (50-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        vignette_scale = pick(vignette_scale, 0.5, 1.0)
        glare = pick(glare, 0.25, 0.6)
        light, _ = light_dark(img)

        # 1. Виньетирование
        kernel_x = cv2.getGaussianKernel(w, w * vignette_scale)
        kernel_y = cv2.getGaussianKernel(h, h * vignette_scale)
        mask = kernel_y @ kernel_x.T
        mask = mask / mask.max()
        img = img * mask[..., np.newaxis]

        # 2. Блик: размытое эллиптическое пятно, тянет к светлому цвету бумаги
        cx, cy = np.random.uniform(0.2, 0.8) * w, np.random.uniform(0.2, 0.8) * h
        sx, sy = np.random.uniform(0.08, 0.25) * w, np.random.uniform(0.08, 0.25) * h
        xx, yy = coords(h, w)
        g = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
        img = img + (light - img) * (g * glare)[..., None]

        # 3. JPEG
        return finish(img, q, 50, 85)
