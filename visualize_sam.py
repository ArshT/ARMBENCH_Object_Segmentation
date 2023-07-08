import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2 as cv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Instances
from detectron2.structures import Boxes




import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
from torchvision.utils import draw_segmentation_masks
from PIL import Image

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor





img_name = "0A0WG84517.jpg"
# img_name = "ZZTS6DAJB4.jpg"
# img_name = "K7Q4SNOEB4.jpg"

#################################################################################################
ann_file = os.path.join("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/train.json")
coco = COCO(ann_file)

for i in range(len(coco.getImgIds())):
    img_id = i
    img_obj = coco.loadImgs(img_id)[0]
    anns_obj = coco.loadAnns(coco.getAnnIds(img_id))

    if img_obj["file_name"] == img_name:
        break

img_name = "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/" + img_obj["file_name"]
img = cv.imread(img_name)

torch_img = torch.Tensor(img).type(torch.uint8)
torch_img = torch_img.permute(2, 0, 1)

bboxes = [ann['bbox'] for ann in anns_obj]
masks = [coco.annToMask(ann) for ann in anns_obj]
areas = [ann['area'] for ann in anns_obj]

boxes = torch.tensor(np.array(bboxes), dtype=torch.float32)
boxes = torchvision.ops.box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")


labels = [coco.loadCats(annotation['category_id'])[0]['name'] for annotation in anns_obj]
category_mapping = {"Tote":0,"Object":1}
label_numbers = []
for label in labels:
    label_numbers.append(category_mapping[label])
labels = torch.tensor(label_numbers, dtype=torch.int64)

masks = torch.tensor(np.array(masks), dtype=torch.uint8)
image_id = torch.tensor([i])
area = torch.as_tensor(np.array(areas))
iscrowd = torch.zeros(len(anns_obj), dtype=torch.int64)

target = {}
target["boxes"] = boxes
target["labels"] = labels
target["masks"] = masks
target["image_id"] = image_id
target["area"] = area
target["iscrowd"] = iscrowd

image_with_boxes = draw_bounding_boxes(torch_img,boxes=target["boxes"], width=4)
image_with_seg = draw_segmentation_masks(torch_img, masks=target["masks"].bool(), alpha=0.7)
np_image_with_boxes = image_with_boxes.int().permute(1, 2,0).numpy()
np_image_with_seg = image_with_seg.int().permute(1, 2,0).numpy()
#########################################################################################




def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



sam_checkpoint = "epoch-000004-f10.94-ckpt.pth"
model_type = "vit_b"

# device = "cuda"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor_sam = SamPredictor(sam)

bboxes = target["boxes"]
bboxes = torch.tensor(bboxes, device='cpu')

print()
print("Image Shape",img.shape)

predictor_sam.set_image(img)
transformed_boxes = predictor_sam.transform.apply_boxes_torch(bboxes, img.shape[:2])

masks, _, _ = predictor_sam.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=True,
)

mask_number = 0

masks = masks[:,mask_number,:,:]
masks_numpy = masks.cpu().numpy()
bboxes_numpy = bboxes.cpu().numpy()

print("Masks Results:",masks_numpy.shape, bboxes_numpy.shape)
print()
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for mask in masks_numpy:
    show_mask(mask, plt.gca(), random_color=True)
for box in bboxes_numpy:
    show_box(box, plt.gca())
plt.axis('off')
plt.show()