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
import torch.nn as nn

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
from detectron2.structures import Instances
from detectron2.structures import Boxes
from pycocotools.coco import COCO

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


register_coco_instances("my_dataset_train", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/train.json", "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/")
register_coco_instances("my_dataset_val", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/val.json", "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/")
register_coco_instances("my_dataset_test", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/test.json", "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/images/")

# register_coco_instances("my_dataset_train_same", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/train.json", "/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/images/")
# register_coco_instances("my_dataset_val_same", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/val.json", "/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/images/")
# register_coco_instances("my_dataset_test_same", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/test.json", "/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/images/")

# register_coco_instances("my_dataset_train_zoomed", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/train.json", "/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/images/")
# register_coco_instances("my_dataset_val_zoomed", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/val.json", "/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/images/")
# register_coco_instances("my_dataset_test_zoomed", {}, "/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/test.json", "/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/images/")

dataset_metadata = MetadataCatalog.get("my_dataset_train")
# dataset_metadata = MetadataCatalog.get("my_dataset_train_same")
print("Train Metadata",dataset_metadata)



cfg = get_cfg()


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
    def __init__(self, sam, predictor,coco_object):
    # def __init__(self, sam, coco_object):
        super().__init__()
        self.sam = sam
        self.predictor = predictor
        self.coco = coco_object
        self.max_objects = 0
    
    def forward(self, input):
        try:
            img1 = cv.imread(input[0]["file_name"])
        except:
            img1 = input
        
        outputs = self.predictor(img1)

        bboxes = outputs['instances'].pred_boxes
        classes = outputs['instances'].pred_classes
        scores = outputs['instances'].scores
        
        scores = scores.cpu().numpy()
        scores = list(scores)
        
        scores_list = list()
        bboxes_list = list()
        class_list = list()
        for i in range(len(scores)):
            if scores[i] > 0.75:
                bboxes_list.append(bboxes[i].tensor.cpu().numpy())
                class_list.append(classes[i].cpu().numpy())
                scores_list.append(scores[i])
        
        
        bboxes = torch.tensor(np.array(bboxes_list)).reshape((len(bboxes_list),4))
        classes = torch.tensor(np.array(class_list))
        scores = torch.tensor(np.array(scores_list))

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

        new_instances.pred_masks = masks[:,mask_number,:,:].to(device="cpu")
        new_instances.pred_boxes = Boxes(bboxes).to(device="cpu")
        new_instances.scores = scores
        new_instances.pred_classes = classes
        
        return [{"instances":new_instances}]



class InstanceSegmenterSame(nn.Module):
    def __init__(self, sam, predictor,coco_object):
    # def __init__(self, sam, coco_object):
        super().__init__()
        self.sam = sam
        self.predictor = predictor
        self.coco = coco_object
        self.max_objects = 0
    
    def forward(self, input):
        try:
            img1 = cv.imread(input[0]["file_name"])
        except:
            img1 = input
        
        outputs = self.predictor(img1)

        bboxes = outputs['instances'].pred_boxes
        classes = outputs['instances'].pred_classes
        scores = outputs['instances'].scores
        
        scores = scores.cpu().numpy()
        scores = list(scores)
        
        scores_list = list()
        bboxes_list = list()
        class_list = list()

        for i in range(len(scores)):
            if scores[i] > 0.6:
                if classes[i].cpu().numpy() == 1:
                    bboxes_list.append(bboxes[i].tensor.cpu().numpy())
                    class_list.append(np.array(0))
                    scores_list.append(scores[i])
        
    
        bboxes = torch.tensor(np.array(bboxes_list)).reshape((len(bboxes_list),4))
        classes = torch.tensor(np.array(class_list))
        scores = torch.tensor(np.array(scores_list))

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

        new_instances.pred_masks = masks[:,mask_number,:,:].to(device="cpu")
        new_instances.pred_boxes = Boxes(bboxes).to(device="cpu")
        new_instances.scores = scores
        new_instances.pred_classes = classes
        
        return [{"instances":new_instances}]



train_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/train.json")
val_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/val.json")
test_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/test.json")

# train_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/train.json")
# val_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/val.json")
# test_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/same-object-transfer-set/test.json")

# train_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/train.json")
# val_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/val.json")
# test_coco = COCO("/home/arsh/Desktop/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/test.json")



instance_segmenter = InstanceSegmenter(predictor_sam, predictor,test_coco)
# instance_segmenter = InstanceSegmenterSame(predictor_sam, predictor,test_coco)



def log_results(cfg,instance_segmenter,dataset_name,output_dir):
    setup_logger()
    logger = logging.getLogger(__name__)

    with open('log_sam.txt', 'w') as f:
        with redirect_stdout(f):
            evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
            val_loader = build_detection_test_loader(cfg, dataset_name)
            print("Start")
            results = inference_on_dataset(instance_segmenter, val_loader, evaluator)
            print(results)
            print("Done")
            print()

    logger.info(results)

    return results

def log_results_test(cfg,instance_segmenter,dataset_name,output_dir):
    setup_logger()
    logger = logging.getLogger(__name__)

    with open('log_sam_test_set.txt', 'w') as f:
        with redirect_stdout(f):
            evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
            test_loader = build_detection_test_loader(cfg, dataset_name)
            print("Start")
            results = inference_on_dataset(instance_segmenter, test_loader, evaluator)
            print(results)
            print("Done")
            print()

    logger.info(results)

    return results

# Results
# res = log_results(cfg,instance_segmenter,"my_dataset_val","./output")
res = log_results_test(cfg,instance_segmenter,"my_dataset_test","./output")
print(res)