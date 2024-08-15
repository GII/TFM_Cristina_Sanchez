import os
from pathlib import Path
import cv2
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
import torch
import supervision as sv
from PIL import Image
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

HOME = os.getcwd()

model=sys.argv[1]
print("Utilizando modelo", model)

if model == "sam":
    checkpoint_path = "sam_weights/sam_vit_h_4b8939.pth"
    model_cfg_path = "vit_h"
    sam = sam_model_registry[model_cfg_path](checkpoint=checkpoint_path)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

elif model == "sam2":
    checkpoint_path = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg_path = "sam2_hiera_l.yaml"
    sam = build_sam2(model_cfg_path, checkpoint_path, device ='cuda', apply_postprocessing=True)
    mask_generator = SAM2AutomaticMaskGenerator(sam)

IMAGES_PATH = os.path.join(HOME, "images", "rosa")
for img in Path(IMAGES_PATH).iterdir():
    IMAGE_PATH = os.path.join(IMAGES_PATH, img)
    print(IMAGE_PATH)

    print("Leer imagen")
    image = cv2.imread(IMAGE_PATH)

    print("Generar mascaras")
    sam_result = mask_generator.generate(image)

    print("Numero de mascaras", len(sam_result))

    areas = sorted([result["area"] for result in sam_result])
    print("areas", areas)
    filtered_res = [result for result in sam_result if 300 < result["area"] < 13000]

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    print("Mapear detecciones")
    detections = sv.Detections.from_sam(sam_result=sam_result)

    print("Anotar mascaras")
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    # Clear CUDA cache
    torch.cuda.empty_cache()