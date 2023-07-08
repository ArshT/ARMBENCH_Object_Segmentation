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
import logging
from contextlib import redirect_stdout

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



register_coco_instances("my_dataset_train", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/train.json", "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/")
register_coco_instances("my_dataset_val", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/val.json", "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/")

dataset_metadata = MetadataCatalog.get("my_dataset_train")
print("Train Metadata:",dataset_metadata)

cfg = get_cfg()


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

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor_sam = SamPredictor(sam)

mask_number = 0

mask_limit = 100

class InstanceSegmenter(nn.Module):
    def __init__(self, sam, coco_object):
        super().__init__()
        self.sam = sam
        self.coco = coco_object
        self.max_objects = 0
    
    def forward(self, input):
        anns_obj = self.coco.loadAnns(self.coco.getAnnIds(input[0]["image_id"]))

        img1 = cv.imread(input[0]["file_name"])
        bboxes = [ann['bbox'] for ann in anns_obj]
        bboxes = torch.tensor(bboxes)
        bboxes = bboxes[:mask_limit,:] # mask_limit is the max number of objects in the dataset for memory reasons (Might not be necessary with 12GB VRAM GPU)
        bboxes = torchvision.ops.box_convert(bboxes, in_fmt="xywh", out_fmt="xyxy")

        
        self.sam.set_image(img1)
        transformed_boxes = self.sam.transform.apply_boxes_torch(bboxes, img1.shape[:2])
        transformed_boxes = transformed_boxes.to(device="cuda")
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )

        new_instances = Instances(image_size=(img1.shape[0], img1.shape[1]))
        scores = torch.ones(masks.shape[0])
        
        labels = [self.coco.loadCats(annotation['category_id'])[0]['name'] for annotation in anns_obj][:mask_limit]
        category_mapping = {"Tote":0,"Object":1}
        label_numbers = []
        for label in labels:
            label_numbers.append(category_mapping[label])
        labels = torch.tensor(label_numbers, dtype=torch.int64)

        if self.max_objects < labels.shape[0] - 1:
            self.max_objects = labels.shape[0] - 1

        new_instances.pred_masks = masks[:,mask_number,:,:].to(device="cpu")
        new_instances.pred_boxes = Boxes(bboxes).to(device="cpu")
        new_instances.scores = scores.to(device="cpu")
        new_instances.pred_classes = labels.to(device="cpu")
        
        return [{"instances":new_instances}]


train_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/train.json")
val_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/val.json")
test_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/test.json")


instance_segmenter = InstanceSegmenter(predictor_sam,val_coco)


def log_results(cfg,instance_segmenter,dataset_name,output_dir):
    setup_logger()
    logger = logging.getLogger(__name__)

    with open('log_gt_bboxes.txt', 'w') as f:
        with redirect_stdout(f):
            evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
            val_loader = build_detection_test_loader(cfg, "my_dataset_val")
            print("Start")
            results = inference_on_dataset(instance_segmenter, val_loader, evaluator)
            print(results)
            print("Done")
            print()

    logger.info(results)

    return results


# Results
res = log_results(cfg,instance_segmenter,"my_dataset_val","./output")
print(res)