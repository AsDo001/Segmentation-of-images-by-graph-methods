# Сегментация изображений графовыми методами

Проект реализует два алгоритма сегментации изображений:
1. Метод Felzenszwalb-Huttenlocher
2. Алгоритм на основе минимального покрывающего дерева (Geodesic MST)

## Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/Segmentation-of-images-by-graph-methods.git
cd Segmentation-of-images-by-graph-methods
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Запуск алгоритмов:
```bash
# Felzenszwalb-Huttenlocher
cd Felzenszwalb-Huttenlocher
python Felzenszwalb-Huttenlocher.py

# Geodesic MST
cd ../Geodesic-MST
python Geodesic-MST.py
```

Требования
Python 3.7+

OpenCV

scikit-image

SciPy

NumPy

Matplotlib

## Настройка параметров
В каждом файле .py можно настроить параметры:
```bash
# Для Felzenszwalb-Huttenlocher
segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)

# Для Geodesic MST
max_distance_threshold = mean_dist + 1.5 * std_dist  # Параметр кластеризации
```

