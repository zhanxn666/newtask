# train.py
import ultralytics.nn.tasks as u_tasks
from torch import nn
import torch
import yaml
from argparse import Namespace
# from torch.utils.data import DataLoader
from module.MFFN_YOLO_Backbone import MFFN_YOLO_Backbone
from ultralytics import YOLO
from dataset.mffn_yolo_dataset import MFFN_YOLO_Dataset
from utils.collate import mffn_yolo_collate_fn
import os
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# trainer.py
from ultralytics.models.yolo.detect import DetectionTrainer
from torch.utils.data import DataLoader

from ultralytics.models.yolo.detect.val import DetectionValidator
import torch
from ultralytics.utils import LOGGER, RANK, nms, ops

class MyTrainer(DetectionTrainer):

    def __init__(self, train_dataset, val_dataset,custom_model, cfg, overrides=None, _callbacks=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Ensure overrides is always a dictionary
        if overrides is None:
            overrides = {}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        # 强制Trainer使用你的自定义模型（覆盖默认模型）
        self.model = custom_model.model  # custom_model是YOLO实例，其.model属性是实际的nn.Module
        self.model.to(self.device)  # 确保模型移至GPU/CPU
        
    # --------------------------------------------
    # Override YOLO's default dataloader
    # --------------------------------------------
    def get_dataloader(self, split, **kwargs):
        if split == "train":
            return InfiniteDataLoader(
                self.train_dataset,
                batch_size=kwargs.get('batch_size', 4),  # use YOLO batch_size if provided
                shuffle=kwargs.get('shuffle', True),
                num_workers=kwargs.get('num_workers', 8),
                collate_fn=mffn_yolo_collate_fn,
                # pin_memory=kwargs.get('pin_memory', True),
                pin_memory=False,
                multiprocessing_context='spawn',

            )
        else:  # validation
            return InfiniteDataLoader(
                self.val_dataset,
                batch_size=kwargs.get('batch_size', 4),
                shuffle=kwargs.get('shuffle', False),
                num_workers=kwargs.get('num_workers', 8),
                collate_fn=mffn_yolo_collate_fn,
                # pin_memory=kwargs.get('pin_memory', True),
                pin_memory=False,
                multiprocessing_context='spawn',
            )

        
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # 1. Load powerdata.yaml  (contains path, train/val splits, classes)
    # ----------------------------------------------------------------------
    DATA_YAML = "/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower.yaml"
    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    ROOT_IMGS = data_cfg["path"]                     # /home/.../powerdata/images
    TRAIN_IMG_DIR   = f"{ROOT_IMGS}/{data_cfg['train']}"
    VAL_IMG_DIR     = f"{ROOT_IMGS}/{data_cfg['val']}"
    TRAIN_LABEL_DIR = TRAIN_IMG_DIR.replace("images", "labels")
    VAL_LABEL_DIR   = VAL_IMG_DIR.replace("images", "labels")
    IMGSZ = 384


    # ----------------------------------------------------------------------
    # 2. Build datasets
    # ----------------------------------------------------------------------
    train_dataset = MFFN_YOLO_Dataset(
        root=[
        (ROOT_IMGS, {"image": {"path": data_cfg["train"], "suffix": ".jpg"}})
        ],
        shape={"h": IMGSZ, "w": IMGSZ},
        label_dir=TRAIN_LABEL_DIR,

    )
    print(f"train_dataset: {train_dataset}")
    val_dataset = MFFN_YOLO_Dataset(
        root=[
            (ROOT_IMGS, {"image": {"path": data_cfg["val"], "suffix": ".jpg"}})
        ],
        shape={"h": IMGSZ, "w": IMGSZ},
        label_dir=VAL_LABEL_DIR,
    )
    print(f"val_dataset: {val_dataset}")
    # ----------------------------------------------------------------------
    # 3. Load model (YAML defines MFFN_YOLO_Backbone + Detect head)
    # ----------------------------------------------------------------------


    # ✅ 注册自定义模块
    u_tasks.MFFN_YOLO_Backbone = MFFN_YOLO_Backbone
    # register Conv2d for YAML parsing
    #u_tasks.Conv2d = nn.Conv2d
    #u_tasks.MFFN_YOLO_Backbone = MFFN_YOLO_Backbone



    # Get directory of train.py
    CFG_DIR = os.path.dirname(__file__)  # or os.path.dirname(os.path.abspath(__file__))

    MODEL_CFG = os.path.join(CFG_DIR, "cfg", "yolo11_mffn2.yaml")  # ← CORRECT PATH TO YAML

    

    # 然后用你的自定义 YAML 替换 backbone（head 保持预训练）
    model = YOLO(MODEL_CFG)  # 你的 yolo11_mffn2.yaml
    model = model.load("yolo11n.pt")  # 加载预训练权重

    # model.model.load = False        # ← stop loading pretrained
    # model.overrides['pretrained'] = False
    
    # Add this after model creation
    print("Model configuration:")
    print(model.model.yaml)  # This shows the actual config being used

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {torch.cuda.get_device_name(device) if device != 'cpu' else 'CPU'}")

    # ----------------------------------------------------------------------
    # 4. Train – Ultralytics accepts custom loaders
    # ----------------------------------------------------------------------

    # 替换所有打印代码，这是最终稳定版
    print("\n" + "="*80)
    print("📋 自定义 MFFN-YOLO 模型简洁结构摘要")
    print("="*80)

    # 核心：遍历 YOLO 模型的顶层模块（backbone + head）
    for i, (name, module) in enumerate(model.model.model.named_children()):
        # 只打印模块名、类名、参数量（关键信息）
        params = sum(p.numel() for p in module.parameters())  # 计算模块参数量
        print(f"{i:2d} 模块名：{name:15s} 类名：{module.__class__.__name__:20s} 参数量：{params:,}")

    # 打印模型总统计（复用 YOLO 原生 info 方法）
    print(f"\n📊 模型总统计：{model.model.info()}")
    
    trainer = MyTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg="/home/e706/zhanxiangning/newtask/yolo_mffn/cfg/cfg.yaml",
        custom_model=model,
        overrides={},
    )

    trainer.train()

   