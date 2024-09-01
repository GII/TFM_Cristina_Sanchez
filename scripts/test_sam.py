import os
from pathlib import Path
import time
import cv2
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
import torch
import supervision as sv
from torch.amp import autocast
import sys
import evaluation

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

HOME = os.getcwd()

model=sys.argv[1]
print("Utilizando modelo", model)

if model == "sam":
    checkpoint_path = "sam_weights/sam_vit_h_4b8939.pth"
    model_cfg_path = "vit_h"
    sam = sam_model_registry[model_cfg_path](checkpoint=checkpoint_path).half()
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam,  points_per_side=56, pred_iou_thresh=0.95, stability_score_thresh=0.96)

elif model == "sam2":
    checkpoint_path = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg_path = "sam2_hiera_l.yaml"
    sam = build_sam2(model_cfg_path, checkpoint_path, device ='cuda', apply_postprocessing=True).half()
    mask_generator = SAM2AutomaticMaskGenerator(sam,  points_per_side=64, pred_iou_thresh=0.76, stability_score_thresh=0.93)

IMAGES_PATH = os.path.join(HOME, "images")
for img in Path(IMAGES_PATH).iterdir():
    IMAGE_PATH = os.path.join(IMAGES_PATH, img)
    print(IMAGE_PATH)

    print("Leer imagen")
    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (780,780))
    
    print("Generar mascaras")
    start_time = time.time()
    with autocast(device_type="cuda"):
        sam_result = mask_generator.generate(image)
    stop_time= time.time()
    print("Tiempo de procesamiento:", stop_time-start_time)
    print("Numero de mascaras", len(sam_result))

    areas = sorted([result["area"] for result in sam_result])
    print("areas", areas)
    filtered_res = [result for result in sam_result if 100 < result["area"] < 950]

    mask_annotator = sv.MaskAnnotator(opacity=1, color_lookup=sv.ColorLookup.INDEX)

    print("Mapear detecciones")
    detections = sv.Detections.from_sam(sam_result=filtered_res)

    print("Anotar mascaras")
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    evaluate = sys.argv[2] if len(sys.argv)>2 else None
    if evaluate:
        tp = int(input("Cuantos Verdaderos Positivos (VP) hay? "))
        fp = int(input("Cuantos Falsos Positivos (FP) hay? "))
        fn = int(input("Cuantos Falsos Negativos (FN) hay? "))

        precision = evaluation.calculate_precision(tp,fp)
        recall = evaluation.calculate_recall(tp,fp,fn)
        f1 = evaluation.calculate_f1_score(precision, recall)
        
        print("Precision", precision)
        print("Recall", recall)
        print("F1 Score", f1)
        