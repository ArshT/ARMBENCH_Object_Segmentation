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
import torchvision

import cv2 as cv
import os
import sys
import itertools
import matplotlib.pyplot as plt
from PIL import Image


import sys
from segment_anything import sam_model_registry, SamPredictor

# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
from typing import Any, Dict, List, Set

import torch

import logging
from contextlib import redirect_stdout
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from d2.detr import DetrDatasetMapper, add_detr_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data.datasets import register_coco_instances
from detectron2.solver.build import maybe_add_gradient_clipping



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




class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if "Detr" == cfg.MODEL.META_ARCHITECTURE:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer



cfg = get_cfg()

def setup(args):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.merge_from_file('configs/detr_256_6_6_torchvision.yaml')
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

args = default_argument_parser().parse_args()
cfg = setup(args)
predictor = DefaultPredictor(cfg)
model = Trainer.build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)



img_name = "0A0WG84517.jpg"
# img_name = "ZZTS6DAJB4.jpg"
# img_name = "K7Q4SNOEB4.jpg"
# img_name = "0A2ZTW5KDG.jpg"
# img_name = "0ANDW9OLG8.jpg"

img_name = "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/" + img_name
img = cv.imread(img_name)
print("Image Shape:",img.shape)

some_op = predictor(img)
# print(some_op)

bboxes = some_op['instances'].pred_boxes
classes = some_op['instances'].pred_classes

scores = some_op['instances'].scores
scores = scores.cpu().numpy()
scores = list(scores)


bboxes_list = list()
class_list = list()
for i in range(len(scores)):
    if scores[i] > 0.75:
        bboxes_list.append(bboxes[i].tensor.cpu().numpy())
        class_list.append(classes[i].cpu().numpy())


bboxes = torch.tensor(np.array(bboxes_list))


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img,boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
 
    for box, c in zip(boxes.tolist(), COLORS * 100):
        xmin, ymin, xmax, ymax = box[0]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))

    plt.axis('off')
    plt.show()


im = Image.open(img_name)
plot_results(im, bboxes)


sam_checkpoint = "epoch-000004-f10.94-ckpt.pth"
model_type = "vit_b"


# device = "cuda"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor_sam = SamPredictor(sam)

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


plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for mask in masks_numpy:
    show_mask(mask, plt.gca(), random_color=True)
for box in bboxes_numpy:
    show_box(box.reshape(-1), plt.gca())
plt.axis('off')
plt.show()
