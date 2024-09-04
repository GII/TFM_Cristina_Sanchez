import os
from pathlib import Path
import time
import cv2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
import torch
import supervision as sv
from torch.amp import autocast


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

HOME = os.getcwd()
IMAGES_PATH = os.path.join(HOME, "images")

checkpoint_path = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg_path = "sam2_hiera_l.yaml"
sam = build_sam2(model_cfg_path, checkpoint_path, device ='cuda', apply_postprocessing=True).half()
mask_generator = SAM2AutomaticMaskGenerator(sam, points_per_side=64, pred_iou_thresh=0.76, stability_score_thresh=0.93)

MAX_N_VIRUTAS = 195
MIN_N_VIRUTAS = 135

MAX_AREA_VIRUTA = 1050
MIN_AREA_VIRUTA = 100

AREA_DONUT_APROXIMADA = 447971

def segmentar_imagen(img):
    print("Leer imagen")
    IMAGE_PATH = os.path.join(IMAGES_PATH, img)
    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (780,780))

    print("Generar mascaras")
    start_time = time.time()
    with autocast(device_type="cuda"):
        sam_result = mask_generator.generate(image)
    stop_time= time.time()
    tiempo_procesamiento = stop_time-start_time
    print("Tiempo de procesamiento:", tiempo_procesamiento)
    print("Numero de mascaras total", len(sam_result), "\n")

    print(" Procesando Donut  \n")
    print(sorted([viruta["area"] for viruta in sam_result]))
    filtered_res = [result for result in sam_result if MIN_AREA_VIRUTA < result["area"] < MAX_AREA_VIRUTA]
    numero_virutas = len(filtered_res)
    area_total_virutas = sum(viruta["area"] for viruta in filtered_res)
    
    valido = False
    if not MIN_N_VIRUTAS < numero_virutas < MAX_N_VIRUTAS:
        mensaje = f"""DONUT NO VALIDO! EL NUMERO DE VIRUTAS DETECTADAS <{numero_virutas}>
        NO ESTÁ EN EL RANGO <{MIN_N_VIRUTAS}:{MAX_N_VIRUTAS}> \n"""
    elif not MIN_AREA_VIRUTA*MIN_N_VIRUTAS < area_total_virutas < MAX_AREA_VIRUTA*MAX_N_VIRUTAS:
        mensaje = f"""DONUT NO VALIDO! EL AREA TOTAL DE LAS VIRUTAS DETECTADAS <{area_total_virutas} px>
        NO ESTÁ EN EL RANGO <{MIN_AREA_VIRUTA*MIN_N_VIRUTAS}:{ MAX_AREA_VIRUTA*MAX_N_VIRUTAS}> \n"""
    elif not 0.1 < area_total_virutas/AREA_DONUT_APROXIMADA < 0.4:
        mensaje = f"""DONUT NO VÁLIDO! Número de virutas {numero_virutas} adecuado, pero con un área total de
                    {area_total_virutas} px \n"""
    else:
        valido = True
        mensaje = f"""DONUT VÁLIDO! Número de virutas {numero_virutas}, con un área con respecto al donut del
               {100*area_total_virutas/AREA_DONUT_APROXIMADA}%\n"""
    print(mensaje)

    mask_annotator = sv.MaskAnnotator(color=sv.Color.BLACK, opacity=1, color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=filtered_res)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    cv2.imwrite('imagen_segmentada.jpg', annotated_image)
    
    # Limpiar CUDA cache
    torch.cuda.empty_cache()
    return numero_virutas, area_total_virutas, tiempo_procesamiento, valido

if __name__ == "__main__":
    for img in Path(IMAGES_PATH).iterdir():
        IMAGE_PATH = os.path.join(IMAGES_PATH, img)
        segmentar_imagen(img)