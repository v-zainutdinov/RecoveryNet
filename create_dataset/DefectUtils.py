"""Общие помощники для файлов с дефектами (EasyDefects / EasyAndMediumDefects /
MediumDefects / HardDefects) и для effects-скрипта датасета.

Все эффекты работают с BGR uint8 (H, W, 3) — как cv2.imread. Важно: эффекты
НЕ меняют цвет/тон снимка (серый остаётся серым, сепия — сепией). Цвета
«светлого» (бумага, царапины) и «тёмного» (пыль, пятна) берутся из самой картинки
через tone_colors(), поэтому один и тот же код работает и на ч/б, и на сепии.
"""
import numpy as np
import cv2


def finish(img: np.ndarray, q, q_min: int, q_max: int) -> np.ndarray:
    """Последний шаг любого эффекта: в uint8 + JPEG.
        q: None - случайное качество из [q_min, q_max];
           0 - без JPEG (нужно для цепочек эффектов, чтобы JPEG не копился);
           число - это качество
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    if q == 0:
        return img
    if q is None:
        q = np.random.randint(q_min, q_max + 1)
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def pick(value, low: float, high: float) -> float:
    """Если параметр не задан (None) — случайное из [low, high], иначе он сам."""
    if value is None:
        return float(np.random.uniform(low, high))
    return float(value)


def pick_int(value, low: int, high: int) -> int:
    """То же для целых (high включительно)."""
    if value is None:
        return int(np.random.randint(low, high + 1))
    return int(value)


def density(h: int, w: int) -> float:
    """Множитель для количества мелких дефектов (пыль, пятнышки): на маленьких
    картинках их должно быть меньше, на больших — больше. За норму берём 512x512."""
    return float(np.clip(h * w / 262144.0, 0.15, 4.0))


def luma(img: np.ndarray) -> np.ndarray:
    """Яркость BGR изображения (H, W), float32."""
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


def coords(h: int, w: int):
    """Сетки координат (xx, yy), float32, размера (h, w)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def tone_colors(img: np.ndarray):
    """Цвет «бумаги» (светлые места) и «чернил» (тёмные) этой картинки, BGR (3,)."""
    small = img[::4, ::4].reshape(-1, 3).astype(np.float32)
    paper = np.percentile(small, 97, axis=0)
    ink = np.percentile(small, 3, axis=0)
    return paper, ink


def light_dark(img: np.ndarray):
    """(светлый, тёмный) цвета для дефектов: светлый — почти белая бумага тона
    этого снимка (царапины, сколы), тёмный — тёмная пыль/пятна."""
    paper, ink = tone_colors(img)
    light = paper + (255.0 - paper) * 0.6
    dark = ink * 0.5
    return light.astype(np.float32), dark.astype(np.float32)


def pick_background(light: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Что видно в месте утраты: чаще светлая бумага, реже тёмная/серая основа."""
    u = np.random.rand()
    if u < 0.7:
        return light
    if u < 0.9:
        return dark
    return (light + dark) / 2.0


def smooth_field(h: int, w: int, cells: float) -> np.ndarray:
    """Гладкое низкочастотное случайное поле [0, 1] размера (h, w).
    cells ~ сколько случайных «клеток» по короткой стороне (меньше -> плавнее)."""
    short = min(h, w)
    ch = max(2, int(round(h / short * cells)))
    cw = max(2, int(round(w / short * cells)))
    small = np.random.rand(ch, cw).astype(np.float32)
    f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return (f - f.min()) / (f.max() - f.min() + 1e-6)


def blob_distance(h: int, w: int, center, radius: float,
                  rough: float = 0.9, cells: float = 5) -> np.ndarray:
    """«Расстояние до центра» с рваным краем: d < 1 — внутри пятна/дыры.
    rough — насколько неровный край (0 — идеальный круг). Считается только
    рядом с пятном, вне окна d = 9 (быстро даже на больших картинках)."""
    d = np.full((h, w), 9.0, np.float32)
    pad = radius * (1.4 + rough)
    x0, x1 = int(max(0, center[0] - pad)), int(min(w, center[0] + pad + 1))
    y0, y1 = int(max(0, center[1] - pad)), int(min(h, center[1] + pad + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return d
    ch, cw = y1 - y0, x1 - x0
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dc = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(radius, 1.0)
    dc = dc + (smooth_field(ch, cw, cells) - 0.5) * rough
    dc = dc + (smooth_field(ch, cw, cells * 6) - 0.5) * rough * 0.25
    d[y0:y1, x0:x1] = dc
    return d


def fill_region(img: np.ndarray, d: np.ndarray, color, tex=None,
                rim: float = 0.0, rim_color=None) -> np.ndarray:
    """Заливает область d < 1 цветом color (+ текстура tex), по краю — светлая
    обтрёпанная кайма силой rim. Края мягкие, без ступенек."""
    m = (1.0 - smoothstep(0.93, 1.0, d))[..., None]
    fill = color if tex is None else color + tex[..., None]
    if rim > 0:
        r = (np.exp(-((d - 1.05) / 0.07) ** 2) * rim)[..., None]
        img = img * (1 - r) + (color if rim_color is None else rim_color) * r
    return img * (1 - m) + fill * m


def line_noise(t: np.ndarray, scale: float = 30.0) -> np.ndarray:
    """1D-шум вдоль координаты t (для прерывистых линий), значения [0, 1]."""
    lo, hi = float(t.min()), float(t.max())
    n = max(4, int((hi - lo) / scale) + 2)
    xs = np.linspace(lo, hi, n)
    vs = np.random.rand(n)
    out = np.interp(t.ravel(), xs, vs).reshape(t.shape).astype(np.float32)
    return np.clip(out * 1.8 - 0.4, 0, 1)


def random_walk(start, angle: float, length: float, steps: int, wobble: float):
    """Извилистая линия: список точек, угол слегка «гуляет» на каждом шаге."""
    pts = [start]
    step = length / max(steps, 1)
    x, y = start
    a = angle
    for _ in range(steps):
        a += np.random.normal(0, wobble)
        x += step * np.cos(a)
        y += step * np.sin(a)
        pts.append((x, y))
    return pts


class Overlay:
    """Слой для рисования тонких штрихов (царапины, пыль, волоски).
    Хранит альфу A и премультиплицированный цвет P, в конце смягчает края."""

    def __init__(self, h: int, w: int):
        self.A = np.zeros((h, w), np.float32)
        self.P = np.zeros((h, w, 3), np.float32)

    @staticmethod
    def _c(col, a):
        return tuple(float(v) * float(a) for v in col)

    def line(self, p1, p2, col, a: float, thickness: int = 1):
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.line(self.A, p1, p2, float(a), thickness)
        cv2.line(self.P, p1, p2, self._c(col, a), thickness)

    def polyline(self, pts, col, a: float, thickness: int = 1, gap_prob: float = 0.0):
        for p1, p2 in zip(pts[:-1], pts[1:]):
            if gap_prob > 0 and np.random.rand() < gap_prob:
                continue
            self.line(p1, p2, col, a, thickness)

    def circle(self, c, r: int, col, a: float):
        c = (int(c[0]), int(c[1]))
        cv2.circle(self.A, c, int(r), float(a), -1)
        cv2.circle(self.P, c, int(r), self._c(col, a), -1)

    def apply(self, img: np.ndarray, blur: float = 0.7) -> np.ndarray:
        A = cv2.GaussianBlur(self.A, (0, 0), blur)
        P = cv2.GaussianBlur(self.P, (0, 0), blur)
        return img * (1.0 - A[..., None]) + P


def to_old_tone(img: np.ndarray, sepia=None) -> np.ndarray:
    """Чистое фото -> ч/б или сепия (3 канала). Это ЦЕЛЬ реставрации: модель не
    должна угадывать цвет (колоризация — отдельная задача), только убирать дефекты.
        sepia: True/False или None (случайно 50/50)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if sepia is None:
        sepia = np.random.rand() < 0.5
    if not sepia:
        return np.clip(np.repeat(gray[..., None] * 255.0, 3, axis=2), 0, 255).astype(np.uint8)
    ink = np.array([22, 38, 55], np.float32) + np.random.uniform(-8, 8, 3)
    paper = np.array([185, 212, 235], np.float32) + np.random.uniform(-10, 10, 3)
    out = ink + gray[..., None] * (paper - ink)
    return np.clip(out, 0, 255).astype(np.uint8)
