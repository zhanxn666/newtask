# train.py
from torch import nn
import torch
import yaml
from argparse import Namespace
from torch.utils.data import DataLoader
from module.MFFN_YOLO_Backbone import MFFN_YOLO_Backbone
from ultralytics import YOLO
from dataset.mffn_yolo_dataset import MFFN_YOLO_Dataset
from utils.collate import mffn_yolo_collate_fn

# ----------------------------------------------------------------------
# 1. Load powerdata.yaml  (contains path, train/val splits, classes)
# ----------------------------------------------------------------------
DATA_YAML = "/home/e706/zhanxiangning/newtask/powerdatasets/powerdata.yaml"
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

ROOT_IMGS = data_cfg["path"]                     # /home/.../powerdata/images
TRAIN_IMG_DIR   = f"{ROOT_IMGS}/{data_cfg['train']}"
VAL_IMG_DIR     = f"{ROOT_IMGS}/{data_cfg['val']}"
TRAIN_LABEL_DIR = TRAIN_IMG_DIR.replace("images", "labels")
VAL_LABEL_DIR   = VAL_IMG_DIR.replace("images", "labels")

# ----------------------------------------------------------------------
# 2. Build datasets
# ----------------------------------------------------------------------
train_dataset = MFFN_YOLO_Dataset(
    root=[
        (ROOT_IMGS, {"image": {"path": data_cfg["train"], "suffix": ".jpg"}})
    ],
    shape={"h": 384, "w": 384},
    label_dir=TRAIN_LABEL_DIR,
)

val_dataset = MFFN_YOLO_Dataset(
    root=[
        (ROOT_IMGS, {"image": {"path": data_cfg["val"], "suffix": ".jpg"}})
    ],
    shape={"h": 384, "w": 384},
    label_dir=VAL_LABEL_DIR,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,                # 5-view → keep small
    shuffle=True,
    num_workers=8,
    collate_fn=mffn_yolo_collate_fn,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=8,
    collate_fn=mffn_yolo_collate_fn,
    pin_memory=True,
)

# ----------------------------------------------------------------------
# 3. Load model (YAML defines MFFN_YOLO_Backbone + Detect head)
# ----------------------------------------------------------------------
import ultralytics.nn.tasks as u_tasks
# ✅ 注册自定义模块
u_tasks.MFFN_YOLO_Backbone = MFFN_YOLO_Backbone
# register Conv2d for YAML parsing
#u_tasks.Conv2d = nn.Conv2d
#u_tasks.MFFN_YOLO_Backbone = MFFN_YOLO_Backbone
import os

# Get directory of train.py
CFG_DIR = os.path.dirname(__file__)  # or os.path.dirname(os.path.abspath(__file__))

MODEL_CFG = os.path.join(CFG_DIR, "cfg", "yolo11_mffn.yaml")  # ← CORRECT PATH TO YAML

model = YOLO(MODEL_CFG)

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Training on: {torch.cuda.get_device_name(device) if device != 'cpu' else 'CPU'}")

# ----------------------------------------------------------------------
# 4. Train – Ultralytics accepts custom loaders
# ----------------------------------------------------------------------
trainer = model.train(
    data=DATA_YAML,
    epochs=1,
    batch=4,
    imgsz=384,
    device=device,
    project="runs/mffn_yolo",
    name="powerdata",
    amp=False
)

# ← INJECT CUSTOM LOADERS
trainer.train_loader = train_loader
trainer.val_loader = val_loader
   