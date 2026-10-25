import numpy as np
import cv2

from src.create_dataset.DefectUtils import (finish, pick, pick_int, density, luma, coords, smoothstep,
                         smooth_field, blob_distance, fill_region, pick_background,
                         line_noise, tone_colors, light_dark, Overlay)
from src.create_dataset.EasyDefects import EasyDefects
from src.create_dataset.EasyAndMediumDefects import EasyAndMediumDefects
from src.create_dataset.MediumDefects import MediumDefects


# Параметры можно не передавать (None) — тогда они берутся случайно из диапазона в docstring.
# q: качество JPEG, 0 - без JPEG (нужно для цепочек эффектов)
class HardDefects:
    def __init__(self, image_array: np.ndarray):
        self.image_array=image_array

    def apply_creases_pipeline(self, n_creases: int=None, width: float=None, displace: float=None, q: int=None) -> np.ndarray: # Сложные
        """Сгибы бумаги: сдвиг пикселей через линию + белая потёртость кусками + тень рядом
            Args:
                n_creases: количество сгибов (1-3)
                width: ширина линии сгиба в пикселях (1.2-3.0)
                displace: сдвиг пикселей поперёк сгиба (1-4)
                q: сжатие JPEG (45-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        s = max(min(h, w) / 512.0, 0.5)
        xx, yy = coords(h, w)
        light, _ = light_dark(img)

        for _ in range(pick_int(n_creases, 1, 3)):
            cx = np.random.uniform(0.2 * w, 0.8 * w)
            cy = np.random.uniform(0.2 * h, 0.8 * h)
            theta = np.random.uniform(0, np.pi)
            nx, ny = -np.sin(theta), np.cos(theta)   # нормаль к сгибу
            tx, ty = np.cos(theta), np.sin(theta)    # направление сгиба

            # 1. Расстояние до слегка кривой линии и координата вдоль неё
            d = (xx - cx) * nx + (yy - cy) * ny
            t = (xx - cx) * tx + (yy - cy) * ty
            d = d + 1.5 * np.sin(t * 0.02 + np.random.uniform(0, 6.28)) + 0.8 * np.sin(t * 0.07 + np.random.uniform(0, 6.28))

            # 2. Картинка «ломается»: по разные стороны сгиба сдвиг в противоположные стороны
            wd = pick(width, 1.2, 3.0) * s
            dis = pick(displace, 1, 4) * s
            shift = dis * np.tanh(d / wd) * np.exp(-(d / (wd * 12)) ** 2)
            img = cv2.remap(img, (xx - nx * shift).astype(np.float32), (yy - ny * shift).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # 3. Белая потёртость идёт кусками, рядом - тень от изгиба
            mod = line_noise(t, 30.0)
            hl = np.clip(np.exp(-(d / wd) ** 2) * np.random.uniform(0.4, 0.85) * mod, 0, 1)[..., None]
            sh = (np.exp(-((d - wd * 2.5) / (wd * 2.0)) ** 2) * np.random.uniform(0.1, 0.3) * (0.4 + 0.6 * mod))[..., None]
            img = img * (1 - sh)
            img = img * (1 - hl) + light * hl

        # 4. JPEG
        return finish(img, q, 45, 85)

    def apply_torn_edges_pipeline(self, n_chips: int=None, size: float=None, corner_cut: float=0.5, q: int=None) -> np.ndarray: # Сложные
        """Рваные края: отколы вдоль краёв кадра и оторванный угол
            Args:
                n_chips: количество отколов (2-6)
                size: радиус откола, доля короткой стороны (0.03-0.1)
                corner_cut: вероятность оторванного угла (0-1)
                q: сжатие JPEG (45-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        m = min(h, w)
        xx, yy = coords(h, w)
        light, dark = light_dark(img)
        tex = (smooth_field(h, w, 40) - 0.5) * 14   # текстура подложки

        # 1. Отколы вдоль краёв (центр чуть за краем кадра)
        for _ in range(pick_int(n_chips, 2, 6)):
            r = pick(size, 0.03, 0.1) * m
            side = np.random.randint(4)
            if side == 0:
                c = (np.random.uniform(0, w), -0.1 * r)
            elif side == 1:
                c = (np.random.uniform(0, w), h - 1 + 0.1 * r)
            elif side == 2:
                c = (-0.1 * r, np.random.uniform(0, h))
            else:
                c = (w - 1 + 0.1 * r, np.random.uniform(0, h))
            d = blob_distance(h, w, c, r, rough=1.0, cells=6)
            img = fill_region(img, d, pick_background(light, dark), tex, rim=0.5, rim_color=light)

        # 2. Оторванный угол: диагональный срез с рваной границей
        if np.random.rand() < corner_cut:
            size_c = np.random.uniform(0.05, 0.18) * m
            fx = xx if np.random.rand() < 0.5 else (w - 1 - xx)
            fy = yy if np.random.rand() < 0.5 else (h - 1 - yy)
            d = (fx + fy) / size_c + (smooth_field(h, w, 6) - 0.5) * 0.7 + (smooth_field(h, w, 40) - 0.5) * 0.15
            img = fill_region(img, d, pick_background(light, dark), tex, rim=0.5, rim_color=light)

        # 3. JPEG
        return finish(img, q, 45, 85)

    def apply_emulsion_loss_pipeline(self, n_chips: int=None, size: float=None, q: int=None) -> np.ndarray: # Сложные
        """Утрата эмульсии: небольшие белые сколы внутри кадра с обтрёпанным краем
            Args:
                n_chips: количество сколов (4-14)
                size: радиус скола, доля короткой стороны (0.008-0.035)
                q: сжатие JPEG (45-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        m = min(h, w)
        light, dark = light_dark(img)
        tex = (smooth_field(h, w, 40) - 0.5) * 14

        for _ in range(pick_int(n_chips, 4, 14)):
            r = max(2.0, pick(size, 0.008, 0.035) * m)
            c = (np.random.uniform(0, w), np.random.uniform(0, h))

            # 1. Скол: чаще чисто белая бумага, реже серая основа
            d = blob_distance(h, w, c, r, rough=1.0, cells=5)
            color = light if np.random.rand() < 0.85 else (light + dark) / 2.0
            img = fill_region(img, d, color, tex, rim=0.4, rim_color=light)

        # 2. JPEG
        return finish(img, q, 45, 85)

    def apply_mold_foxing_pipeline(self, n_blotches: int=None, blotch_size: float=None, n_spots: int=None, strength: float=None, q: int=None) -> np.ndarray: # Сложные
        """Плесень и «лисьи пятна»: много мелких тёмных пятнышек + крупные неровные пятна плесени
            Args:
                n_blotches: количество крупных пятен (1-4)
                blotch_size: радиус крупного пятна, доля короткой стороны (0.04-0.12)
                n_spots: количество мелких пятнышек (60-250 для 512x512)
                strength: непрозрачность пятен (0.3-0.7)
                q: сжатие JPEG (45-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        m = min(h, w)
        s = max(m / 512.0, 0.5)
        _, ink = tone_colors(img)

        strength = pick(strength, 0.3, 0.7)
        n_spots = max(1, int(pick_int(n_spots, 60, 250) * density(h, w)))

        # 1. Мелкие пятнышки
        ov = Overlay(h, w)
        for _ in range(n_spots):
            c = (np.random.randint(0, w), np.random.randint(0, h))
            col = np.clip(ink * np.random.uniform(0.8, 1.5) + 10, 0, 255)
            ov.circle(c, max(1, int(np.random.randint(1, 5) * s)), col, strength * np.random.uniform(0.3, 1.0))
        img = ov.apply(img, blur=1.0)

        # 2. Крупные пятна плесени с внутренней текстурой
        for _ in range(pick_int(n_blotches, 1, 4)):
            r = pick(blotch_size, 0.04, 0.12) * m
            c = (np.random.uniform(0, w), np.random.uniform(0, h))
            d = blob_distance(h, w, c, r, rough=0.9, cells=6)
            inside = 1 - smoothstep(0.7, 1.0, d)
            alpha = np.clip(inside * (0.5 + smooth_field(h, w, 30)) * strength, 0, 1)[..., None]
            col = np.clip(ink * 1.4 + 20, 0, 255)
            img = img * (1 - alpha) + (0.45 * img + 0.55 * col) * alpha

        # 3. JPEG
        return finish(img, q, 45, 85)

    def apply_water_damage_pipeline(self, size: float=None, warp: float=None, blur: float=None, darken: float=None, q: int=None) -> np.ndarray: # Сложные
        """Следы воды: область, где картинка размыта, перекошена и потемнела, по краю ободок
            Args:
                size: радиус области, доля большей стороны (0.25-0.6)
                warp: максимальное смещение пикселей в области (4-12)
                blur: сила размытия в области, сигма (1.5-4.0)
                darken: затемнение области (0.3-0.6)
                q: сжатие JPEG (30-75)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        s = max(min(h, w) / 512.0, 0.5)

        c = (np.random.uniform(0.15 * w, 0.85 * w), np.random.uniform(0.15 * h, 0.85 * h))
        r = pick(size, 0.25, 0.6) * max(h, w) * 0.5
        d = blob_distance(h, w, c, r, rough=0.9, cells=4)
        inside = 1 - smoothstep(0.85, 1.0, d)
        rim = np.exp(-((d - 0.95) / 0.05) ** 2)

        # 1. Перекос (бумагу повело) только внутри области
        wp = pick(warp, 4, 12) * s
        dx = (smooth_field(h, w, 6) - 0.5) * 2 * wp * inside
        dy = (smooth_field(h, w, 6) - 0.5) * 2 * wp * inside
        xx, yy = coords(h, w)
        img = cv2.remap(img, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 2. Размытие (эмульсия поплыла)
        blurred = cv2.GaussianBlur(img, (0, 0), pick(blur, 1.5, 4.0) * s)
        mix = (inside * 0.85)[..., None]
        img = img * (1 - mix) + blurred * mix

        # 3. Потемнение области и тёмный ободок высохшей воды
        dk = pick(darken, 0.3, 0.6)
        img = img * (1 - (inside * dk * 0.35 + rim * 0.25)[..., None])

        # 4. JPEG
        return finish(img, q, 30, 75)

    def apply_craquelure_pipeline(self, n_cells: int=None, crack_width: float=None, coverage: float=None, q: int=None) -> np.ndarray: # Сложные
        """Сетка трещин (кракелюр) на эмульсии: ячейки Вороного с кривыми границами в части кадра
            Args:
                n_cells: количество ячеек (40-120, больше - мельче трещины)
                crack_width: ширина трещины в пикселях (0.8-1.6)
                coverage: доля кадра с трещинами (0.4-1.0)
                q: сжатие JPEG (40-80)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        light, _ = light_dark(img)

        n = pick_int(n_cells, 40, 120)
        cw = pick(crack_width, 0.8, 1.6)
        cov = pick(coverage, 0.4, 1.0)

        # 1. Считаем на уменьшенной копии (до 512 px), потом растягиваем маску
        scale = min(1.0, 512.0 / max(h, w))
        hs, ws = max(8, int(h * scale)), max(8, int(w * scale))
        xx, yy = coords(hs, ws)
        wb = np.random.uniform(4, 10)
        xx = xx + (smooth_field(hs, ws, 8) - 0.5) * 2 * wb
        yy = yy + (smooth_field(hs, ws, 8) - 0.5) * 2 * wb

        # 2. Ближайшая и вторая по близости точка -> границы ячеек
        sx = np.random.uniform(0, ws, n).astype(np.float32)
        sy = np.random.uniform(0, hs, n).astype(np.float32)
        d1 = np.full((hs, ws), np.inf, np.float32)
        d2 = np.full((hs, ws), np.inf, np.float32)
        for i in range(n):
            dd = np.sqrt((xx - sx[i]) ** 2 + (yy - sy[i]) ** 2)
            d2 = np.minimum(d2, np.maximum(d1, dd))
            d1 = np.minimum(d1, dd)

        crack = np.exp(-((d2 - d1) / (cw * scale + 0.3)) ** 2)
        region = smoothstep(1 - cov - 0.15, 1 - cov + 0.15, smooth_field(hs, ws, 3))
        crack = np.clip(cv2.resize(crack * region, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)

        # 3. Тёмная трещина + светлая кромка рядом
        a = np.random.uniform(0.4, 0.8)
        img = img * (1 - (a * crack)[..., None])
        edge = (a * 0.5 * np.roll(crack, 1, axis=1) * (1 - crack))[..., None]
        img = img * (1 - edge) + light * edge

        # 4. JPEG
        return finish(img, q, 40, 80)

    def apply_bad_exposure_pipeline(self, shadows_lift: float=None, highlights_crush: float=None, noise_sigma: float=None, q: int=None) -> np.ndarray: # Сложные
        """Плохая экспозиция скана: выжженные света, поднятые шумные тени, ореолы на контрастных границах
            Args:
                shadows_lift: насколько поднимаем тени (0.1-0.3)
                highlights_crush: порог, выше которого света «выжжены» (0.7-0.9)
                noise_sigma: шум в тенях (8-20)
                q: сжатие JPEG (40-85)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]
        s = max(min(h, w) / 512.0, 0.5)

        shadows_lift = pick(shadows_lift, 0.1, 0.3)
        highlights_crush = pick(highlights_crush, 0.7, 0.9)
        noise_sigma = pick(noise_sigma, 8, 20)

        # 1. Сжимаем dynamic range — света выжжены, тени вытянуты
        img_norm = img / 255.0
        img_norm = img_norm + shadows_lift * (1.0 - img_norm)
        mask = np.clip((img_norm - highlights_crush) / (1.0 - highlights_crush), 0, 1)
        img_norm = img_norm - mask * (img_norm - highlights_crush) * 0.8
        img = img_norm * 255.0

        # 2. Ореолы: тёмные вокруг светлого, светлые вокруг тёмного
        gray = luma(img)
        blur = cv2.GaussianBlur(gray, (0, 0), max(3.0, np.random.uniform(15, 40) * s))
        halo = (gray - blur)[..., None] * np.random.uniform(0.2, 0.5)
        img = img - halo

        # 3. Шум в тенях (усиление теней = усиление шума)
        brightness = luma(img) / 255.0
        shadow_mask = 1.0 - np.clip(brightness * 1.5, 0, 1)
        img = img + np.random.normal(0, noise_sigma, (h, w, 1)) * shadow_mask[..., None]

        # 4. JPEG
        return finish(img, q, 40, 85)

    def apply_heavy_compression_pipeline(self, down_factor: float=None, blur_sigma: float=None, noise_std: float=None, q: int=None) -> np.ndarray: # Сложные
        """Очень плохой скан: низкое разрешение, размытие, шум и жёсткий JPEG
            Args:
                down_factor: во сколько раз уменьшаем (0.3-0.55)
                blur_sigma: гауссово размытие (0.8-2.0)
                noise_std: сила шума (3-10)
                q: сжатие JPEG (8-25)
        """
        img = self.image_array.astype(np.float32)
        h, w = img.shape[:2]

        down_factor = pick(down_factor, 0.3, 0.55)
        blur_sigma = pick(blur_sigma, 0.8, 2.0)
        noise_std = pick(noise_std, 3, 10)

        # 1. Уменьшение и возврат в исходный размер
        small = cv2.resize(img, (max(2, int(w * down_factor)), max(2, int(h * down_factor))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Размытие + шум
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        img = img + np.random.normal(0, noise_std, (h, w, 1)).astype(np.float32)

        # 3. JPEG (жёсткий)
        return finish(img, q, 8, 25)

    def apply_combo_pipeline(self, n_effects: int=None, q: int=None) -> np.ndarray: # Сложные
        """Комбинация 3-4 разных дефектов сразу (как у настоящих старых фото)
            Args:
                n_effects: сколько дефектов наложить (3-4)
                q: сжатие JPEG в конце (20-60)
        """
        n_effects = pick_int(n_effects, 3, 4)

        # Порядок «физический»: сначала то, что с самой бумагой, потом царапины и утраты
        steps = [
            lambda im: EasyDefects(im).apply_film_grain_pipeline(q=0),
            lambda im: MediumDefects(im).apply_uneven_fading_pipeline(q=0),
            lambda im: MediumDefects(im).apply_paper_texture_pipeline(q=0),
            lambda im: MediumDefects(im).apply_glare_vignette_pipeline(q=0),
            lambda im: MediumDefects(im).apply_stains_pipeline(q=0),
            lambda im: HardDefects(im).apply_mold_foxing_pipeline(q=0),
            lambda im: HardDefects(im).apply_water_damage_pipeline(q=0),
            lambda im: HardDefects(im).apply_bad_exposure_pipeline(q=0),
            lambda im: HardDefects(im).apply_creases_pipeline(q=0),
            lambda im: EasyAndMediumDefects(im).apply_scratches_pipeline(q=0),
            lambda im: EasyAndMediumDefects(im).apply_scanner_streaks_pipeline(q=0),
            lambda im: EasyDefects(im).apply_dust_pipeline(q=0),
            lambda im: HardDefects(im).apply_emulsion_loss_pipeline(q=0),
            lambda im: HardDefects(im).apply_torn_edges_pipeline(q=0),
        ]
        idx = sorted(np.random.choice(len(steps), n_effects, replace=False))

        img = self.image_array
        for i in idx:
            img = steps[i](img)

        # JPEG один раз в конце
        return finish(img, q, 20, 60)
