# 🌟 RecoveryNet: Реставрация старых изображений

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()

> **RecoveryNet** — это пайплайн на базе глубокого обучения для комплексного восстановления старых, поврежденных и черно-белых фотографий.

---

## 📸 Результаты (Before / After)

| Оригинальное изображение | После очистки от шумов | После апскейла | Финальная колоризация |
| :---: | :---: | :---: | :---: |
| ![Original](docs/assets/people_001_defect.PNG) | ![Predict](docs/assets/people_001_predict.PNG) | ![Upscaled](assets/upscaled.jpg) | ![Colorized](assets/colorized.jpg) |
| *Поврежденный оригинал* | *Удаление артефактов* | *Повышение разрешения (x4)* | *Реалистичные цвета* |

> 💡 *Нажмите на изображение, чтобы увеличить. Примеры взяты из датасета [Название датасета, например, VOC OldPhotos].*

---

## ⚙️ Архитектура пайплайна

Модель состоит из трех последовательных этапов, каждый из которых оптимизирован для своей задачи:
