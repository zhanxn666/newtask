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

class MyValidator(DetectionValidator):
    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
    
        # Get the class labels for this sample
        cls_tensor = batch["cls"][idx]
    
        # Ensure cls is always 1D, never scalar
        if cls_tensor.ndim == 0:  # scalar tensor
            cls_tensor = cls_tensor.unsqueeze(0)
        elif cls_tensor.ndim > 1:
            cls_tensor = cls_tensor.squeeze(-1)
    
        bbox = batch["bboxes"][idx]
    
        # Same for bboxes - ensure 2D
        if bbox.ndim == 1 and bbox.numel() == 4:
            bbox = bbox.unsqueeze(0)
    
        imgsz = batch["img"].shape[2:]
        bbox_xyxy = ops.xywh2xyxy(bbox)
    
        return {
            "cls": cls_tensor,  # Now guaranteed to be (N,)
            "bboxes": bbox_xyxy,  # Now guaranteed to be (N, 4)
            "ori_shape": imgsz,
            "imgsz": imgsz,
            "ratio_pad": (0, 0, 1, 1),
            "im_file": batch["im_file"][si],
        }
   
    
    # def _process_batch(self, preds, batch):
    #     """对比预测与GT，返回TP/FP匹配情况"""
    #     # preds: List[Tensor] per image, xyxy normalized 0~1
    #     # batch: {"cls":..., "bboxes":..., ...}

    #     pred_cls = preds["cls"]           # (Np,)
    #     pred_boxes = preds["boxes"]       # (Np,4) (xyxy normalized 0~1)

    #     gt_cls = batch["cls"]             # (Ng,)
    #     gt_boxes = batch["bboxes"]        # (Ng,4) xyxy normalized 0~1

    #     if gt_cls.numel() == 0 or pred_cls.numel() == 0:
    #         return {
    #             "tp": torch.zeros(0),
    #             "fp": torch.zeros(0),
    #             "conf": preds["conf"],
    #             "pred_cls": pred_cls,
    #             "gt_cls": gt_cls,
    #         }

    #     # IoU
    #     iou = ops.box_iou(pred_boxes, gt_boxes)  # (Np,Ng)

    #     # 预测匹配条件
    #     iou_threshold = 0.5
    #     pred2gt = iou.max(dim=1)

    #     tp = (pred2gt.values > iou_threshold).float()
    #     fp = 1 - tp

    #     return {
    #         "tp": tp,
    #         "fp": fp,
    #         "conf": preds["conf"],
    #         "pred_cls": pred_cls,
    #         "gt_cls": gt_cls,
    #     }


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

    def preprocess_batch(self, batch):
        """
        Override YOLO preprocess_batch to support 15-channel inputs and GPU transfer.
        """
        # -----------------------------
        # 1. 数据移至GPU（主进程中执行，避免子进程CUDA错误）
        # -----------------------------
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["cls"] = batch["cls"].to(self.device, non_blocking=True)
        batch["bboxes"] = batch["bboxes"].to(self.device, non_blocking=True)
        batch["batch_idx"] = batch["batch_idx"].to(self.device, non_blocking=True)

        # -----------------------------
        # 2. 图像归一化（根据数据集情况调整）
        # -----------------------------
        # 数据集已通过A.Normalize做了减均值/除方差，无需再/255.0；若未做则改为 batch["img"].float() / 255.0
        batch["img"] = batch["img"].float()

        # -----------------------------
        # 3. 验证输入通道（确保是15通道）
        # -----------------------------
        if batch["img"].shape[1] != 15:
            raise ValueError(f"输入通道数错误！预期15，实际{batch['img'].shape[1]}")

        # -----------------------------
        # 4. 标签格式校准
        # -----------------------------
        if "bboxes" in batch:
            batch["bboxes"] = batch["bboxes"].float()
        if "cls" in batch:
            batch["cls"] = batch["cls"].long()
        if "batch_idx" in batch:
            batch["batch_idx"] = batch["batch_idx"].long()

        return batch
        '''
        if isinstance(batch["img"], dict):
            # Example: {'image_c1':Tensor, 'image_o':Tensor, ...}
            for k, v in batch["img"].items():
                # Ensure float in 0~1 (you already normalized in dataset)
                batch["img"][k] = v.float()

        else:
            # Default YOLO behavior
            batch["img"] = batch["img"].float() / 255.0
        
        # -----------------------------
        # 2. Move labels to float
        # -----------------------------
        if "bboxes" in batch:
            batch["bboxes"] = batch["bboxes"].float()

        if "cls" in batch:
            batch["cls"] = batch["cls"].long()

        if "batch_idx" in batch:
            batch["batch_idx"] = batch["batch_idx"].long()

        return batch
        '''
    # --------------------------------------------
    # Override YOLO's default dataloader
    # --------------------------------------------
    def get_dataloader(self, split, **kwargs):
        if split == "train":
            return InfiniteDataLoader(
                self.train_dataset,
                batch_size=8,  # use YOLO batch_size if provided
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
                # batch_size=kwargs.get('batch_size', 4),
                batch_size=8,
                shuffle=kwargs.get('shuffle', False),
                num_workers=kwargs.get('num_workers', 8),
                collate_fn=mffn_yolo_collate_fn,
                # pin_memory=kwargs.get('pin_memory', True),
                pin_memory=False,
                multiprocessing_context='spawn',
            )
    def get_validator(self):
        return MyValidator(self)
    def validate(self):
        # ✔ 强制 validator 使用 test_loader
        self.validator.dataloader = self.test_loader

        # ✔ 调用 validator.run(model)
        metrics = self.validator(self)

        # YOLO 需要返回 (metrics, fitness)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        return metrics, fitness


        
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
    '''
train_dataset = YOLODataset(
    root=[
        (ROOT_IMGS, {"image": {"path": data_cfg["train"], "suffix": ".jpg"}})
    ],
    shape={"h": IMGSZ, "w": IMGSZ},
    label_dir=TRAIN_LABEL_DIR,
)
print(f"train_dataset: {train_dataset}")
val_dataset = YOLODataset(
    root=[
        (ROOT_IMGS, {"image": {"path": data_cfg["val"], "suffix": ".jpg"}})
    ],
    shape={"h": IMGSZ, "w": IMGSZ},
    label_dir=VAL_LABEL_DIR,
)
print(f"val_dataset: {val_dataset}")
    '''
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

    '''
train_loader = DataLoader(
    train_dataset,
    batch_size=4,                # 5-view → keep small
    shuffle=True,
    num_workers=8,
    collate_fn=mffn_yolo_collate_fn,
    pin_memory=True,
)
print(f"train_loader: {train_loader}")
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=8,
    collate_fn=mffn_yolo_collate_fn,
    pin_memory=True,
)
print(f"val_loader: {val_loader}")
    '''
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

    model = YOLO(MODEL_CFG)
    model.model.load = False        # ← stop loading pretrained
    model.overrides['pretrained'] = False

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
'''
from ultralytics.models.yolo.detect import DetectionTrainer

class MyTrainer(DetectionTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode == 'train':
            print("⚡ Using CUSTOM TRAIN DATALOADER!")
            return train_loader
        else:
            print("⚡ Using CUSTOM VAL DATALOADER!")
            return val_loader
trainer = MyTrainer(cfg=None, overrides={})
trainer.model = model
trainer = MyTrainer()
trainer.model = model            # assign your YOLO model
trainer.epochs = 2
trainer.data = DATA_YAML
trainer.imgsz = 640
trainer.device = device
trainer.amp = False
trainer.plots = False
trainer.project = "runs/mffn_yolo"
trainer.name = "publicallpower"
trainer.train()
'''
'''
# ← INJECT CUSTOM LOADERS

model.train_dataloader = lambda: train_loader
model.val_dataloader = lambda: val_loader
#model = YOLO(),YOLO is based on BaseModel, which has train() method, it uses the trainer internally
trainer = model.train(
    data=DATA_YAML,
    epochs=2,
    batch=-1,   # ← Let DataLoader control batch size
    imgsz=640,
    device=device,
    project="runs/mffn_yolo",
    name="publicallpower",
    amp=False,
    plots=False, #this is the problem
)
'''


   