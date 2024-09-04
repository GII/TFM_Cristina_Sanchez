import os
from pathlib import Path
import time
import numpy as np
from skimage import io, color
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb
import cv2
import matplotlib.pyplot as plt


HOME = os.getcwd()

# Cargar la imagen
IMAGES_PATH = os.path.join(HOME, "images", "nuevas")
for img in Path(IMAGES_PATH).iterdir():
    IMAGE_PATH = os.path.join(IMAGES_PATH, img)
    image = io.imread(IMAGE_PATH)

    print("Aplicar la segmentación de Felzenszwalb")
    start_time = time.time()
    segments_fz = felzenszwalb(image, scale=600, sigma=0.75, min_size=100)
    stop_time= time.time()
    print("Tiempo de procesamiento:", stop_time-start_time)

    # Obtener el número de segmentos únicos
    num_segments = segments_fz.max() + 1
    print(num_segments)
    areas = sorted([np.sum(segments_fz == segment_id) for segment_id in range(num_segments) if np.sum(segments_fz == segment_id)])
    print("areas", len(areas), areas)
    
    # Crear una imagen en blanco para colorear las regiones
    colored_segments = np.zeros_like(image)

    # Asignar un color aleatorio a cada segmento
    for segment_id in range(num_segments):
        # Crear un color aleatorio
        color_random = np.random.randint(0, 255, size=3)
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

