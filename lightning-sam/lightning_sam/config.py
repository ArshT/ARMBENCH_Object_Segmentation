from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 6,
    "num_epochs": 6,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 1e-6,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote",
            "annotation_file": "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/train.json"
        },
        "val": {
            "root_dir": "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote",
            "annotation_file": "/home/arsh/Desktop/armbench-segmentation-0.1/mix-object-tote/val.json"
        }
    }
}

cfg = Box(config)
