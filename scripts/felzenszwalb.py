import os
from pathlib import Path
import numpy as np
from skimage import io, color
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb
import cv2
import matplotlib.pyplot as plt


HOME = os.getcwd()

# Cargar la imagen
IMAGES_PATH = os.path.join(HOME, "images", "rosa")
for img in Path(IMAGES_PATH).iterdir():
    IMAGE_PATH = os.path.join(IMAGES_PATH, img)
    image = io.imread(IMAGE_PATH)
    
    # Preprocesamiento
    print("1. Suavizado Gaussiano")
    image = cv2.GaussianBlur(image, (5, 5), 0)

    print("Aplicar la segmentación de Felzenszwalb")
    segments_fz = felzenszwalb(image, scale=700, sigma=0.95, min_size=350)

    # Obtener el número de segmentos únicos
    num_segments = segments_fz.max() + 1
    print(num_segments)
    # Crear una imagen en blanco para colorear las regiones
    colored_segments = np.zeros_like(image)

    # Asignar un color aleatorio a cada segmento
    for segment_id in range(num_segments):
        # Crear un color aleatorio
        color_random = np.random.randint(0, 255, size=3)
        
        # Colorear el segmento con el color aleatorio
        colored_segments[segments_fz == segment_id] = color_random

    # Mostrar la imagen original y la segmentada lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Mostrar la imagen segmentada y coloreada
    axes[1].imshow(colored_segments)
    axes[1].set_title('Segmentación con Felzenszwalb')
    axes[1].axis('off')

    plt.show()

