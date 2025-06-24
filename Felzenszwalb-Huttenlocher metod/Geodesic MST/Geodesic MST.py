import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.future import graph_cut
from skimage import color
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree

#Подгрузка изображения 
img = cv2.imread("Felzenszwalb-Huttenlocher metod\image_before\photo_1_2025-06-24_16-17-16.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lab = color.rgb2lab(img_rgb) #Конвертация в Lab

#Построение суперпиксилей 
superpixels = slic(img_rgb, n_segments=100, compactness=10)

#Построение графа на основе суперпикселей
graph = graph_cut.rag_mean_color(img_rgb, superpixels, mode='similarity')

#Вычисление геодезических расстояний
