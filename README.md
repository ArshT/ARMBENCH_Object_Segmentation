# ARMBENCH Object Segmentation

This repository allows you to use the powerful Segment-Anything model for the Object Segmentation task of the Amazon ARMBENCH Dataset. The respository is built over 
the code from the following repositories:

- The SAM fine-tuning: https://github.com/luca-medeiros/lightning-sam
- The Official DETR Detectron2 Code: https://github.com/facebookresearch/detr/tree/main/d2

Hence, install the dependencies of the above-mentioned repositories.


1. Train a DETR model using the ground-truth bounding boxes from the dataset.
2. Fine-tune the SAM model using Ground-Truth Bounding Boxes as prompts.
3. Evaluate using bounding boxes from DETR as prompts for the finetuned SAM model.

### Running the Code
1. Training a DETR Model:
 
 `cd /DETR/DETR/d2`

`python converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth`

`python train_net.py --config configs/detr_256_6_6_torchvision.yaml --num-gpus 1 MODEL.WEIGHTS converted_model.pth`




2. Finetuning SAM

`cd lightning-sam`

`pip install .`

`python lightning-sam/train.py`



### Results

#### For Mix-Object-Tote task

| Method             |  Mean AP-50  | Mean AP-75 |
| ------------------ | ------------ | ---------- | 
| Mask RCNN FPN with R-50 (Paper)   | 0.72 | 0.61| 
| SAM (Finetuned 4 epochs) + DETR   | **0.86** | **0.76**| 


#### For Same-Object-Tote task

| Method             |  Mean AP-50  | Mean AP-75 |
| ------------------ | ------------ | ---------- | 
| Mask RCNN FPN with R-50 (Paper)   | 0.11 | 0.10 | 
| SAM (Finetuned on Mixed-Object-Tote Dataset for 4 epochs) + DETR   | **0.24** | **0.22**| 

#### For Zoomed-Out-Tote task

| Method             |  Mean AP-50  | Mean AP-75 |
| ------------------ | ------------ | ---------- | 
| Mask RCNN FPN with R-50 (Paper)   | **0.25** | **0.19** | 
| SAM (Finetuned on Mixed-Object-Tote Dataset for 4 epochs) + DETR   | 0.12 | 0.08| 



