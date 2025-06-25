import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, morphology, filters, color, measure, exposure
from skimage.util import img_as_float
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import os

# Загрузка изображения с правильным путем (raw-строка для Windows)
img_path = r'C:\Job\MLE\Segmentation-of-images-by-graph-methods\Geodesic MST\img\photo_1_2025-06-24_16-17-16.jpg'
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")
    
img = io.imread(img_path)
img = img_as_float(img)

# Конвертация с проверкой каналов
if img.ndim == 3 and img.shape[2] == 4:
    img = img[..., :3]  # Удаление альфа-канала
gray_img = color.rgb2gray(img)

# Предварительная обработка

enhanced = exposure.equalize_adapthist(gray_img)
smoothed = filters.median(enhanced, footprint=morphology.disk(3))


thresh = filters.threshold_otsu(smoothed)
markers_img = np.zeros_like(smoothed, dtype=np.int32)
markers_img[smoothed < thresh * 0.7] = 1 
markers_img[smoothed > thresh * 1.3] = 2  


distance = distance_transform_edt(markers_img == 1)
print("Минимальное/максимальное преобразование расстояния:", distance.min(), distance.max())
coords = peak_local_max(distance, footprint=np.ones((3, 3)), min_distance=7, labels=(markers_img == 1))
print("Количество найденных пиков:", len(coords))
if len(coords) == 0:
    raise ValueError("При преобразовании расстояния не обнаружено пиков")
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers_labeled = measure.label(mask)
print("Маркеры, помечающие уникальные значения:", np.unique(markers_labeled))
if len(np.unique(markers_labeled)) <= 1:
    raise ValueError("Нет действительных маркеров, обозначающих водораздел")


segmented = watershed(-distance, markers_labeled, mask=(markers_img != 2))
print("Уникальные значения сегментированного изображения:", np.unique(segmented))
if len(np.unique(segmented)) <= 1:
    raise ValueError("Сегментация водораздела не выявила достоверных регионов")


def create_graph(segmentation_map):
    regions = measure.regionprops(segmentation_map)
    if len(regions) == 0:
        print("Уникальные значения карты сегментации:", np.unique(segmentation_map))
        raise ValueError("На карте сегментации не найдено регионов")
    

    centroids = np.array([r.centroid for r in regions])
    print("Количество регионов:", len(regions))
    print("Форма центроидов:", centroids.shape)
    print("Значения центроидов:", centroids)
    
    if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
        raise ValueError("Центроиды содержат значения NaN или inf")
    
    num_nodes = len(centroids)
    if num_nodes < 2:
        raise ValueError("Для вычисления расстояний требуется как минимум 2 центроида")
    

    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    

    mst = minimum_spanning_tree(csr_matrix(dist_matrix))
    return mst, centroids, segmentation_map

mst, centroids, pre_segmented = create_graph(segmented)


def cluster_mst(mst, max_distance):
    mst_modified = mst.copy()
    mst_modified.data[mst_modified.data > max_distance] = 0
    n_components, labels = connected_components(mst_modified, directed=False)
    return labels


distances = mst.data[~np.isinf(mst.data)]  # Исключаем бесконечные значения
if len(distances) == 0:
    raise ValueError("Нет допустимых расстояний в MST")

mean_dist = np.mean(distances)
std_dist = np.std(distances)
max_distance_threshold = mean_dist + 1.5 * std_dist
cluster_labels = cluster_mst(mst, max_distance=max_distance_threshold)


final_mask = np.zeros_like(pre_segmented, dtype=np.int32)
regions = measure.regionprops(pre_segmented)
for i, region in enumerate(regions):
    final_mask[pre_segmented == region.label] = cluster_labels[i] + 1  

# Отображение результатов
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(img), plt.title('Изображение до преобразования')
plt.subplot(232), plt.imshow(markers_img, cmap='jet'), plt.title('Маркеры')
plt.subplot(233), plt.imshow(distance, cmap='magma'), plt.title('Преобразование расстояния')
plt.subplot(234), plt.imshow(segmented, cmap='nipy_spectral'), plt.title('Сегментация водораздела')
plt.subplot(235), plt.imshow(final_mask, cmap='nipy_spectral'), plt.title('Кластеризация GMST')


plt.subplot(236)
plt.imshow(img)
plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=10)
for i in range(mst.shape[0]):
    for j in range(mst.indptr[i], mst.indptr[i+1]):
        if mst.data[j] > 0 and mst.data[j] <= max_distance_threshold:
            node_i = i
            node_j = mst.indices[j]
            plt.plot([centroids[node_i, 1], centroids[node_j, 1]],
                     [centroids[node_i, 0], centroids[node_j, 0]], 'g-', linewidth=0.5)
plt.title('Центроиды и MST')

plt.tight_layout()
plt.show()


boundaries = segmentation.mark_boundaries(img, final_mask, color=(1, 0, 0))
plt.figure(figsize=(10, 8))
plt.imshow(boundaries)
plt.title('Границы зерен')
plt.show()

# Количественные результаты
print(f"Количество обнаруженных зерен: {len(np.unique(final_mask)) - 1}")
print(f"Пороговое значение расстояния MST: {max_distance_threshold:.2f}")
print(f"Среднее расстояние между центроидами: {mean_dist:.2f}")