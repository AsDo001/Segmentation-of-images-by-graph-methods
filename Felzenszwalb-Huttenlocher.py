import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt

#Метод Felzenszwalb-Huttenlocher
img = cv2.imread("Felzenszwalb-Huttenlocher\origin_img\g1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Преобразование в приемлимый цвет

segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(img)
plt.axis('on')

plt.subplot(1, 2, 2)
plt.title("Изображение после сегментации")
plt.imshow(segments, cmap='nipy_spectral')
plt.axis('on')

plt.show()

# Сохранение результата
cv2.imwrite('felzenszwalb_segmentation7.jpg', cv2.cvtColor(segments.astype(np.uint8), cv2.COLOR_GRAY2BGR))