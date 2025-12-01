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
    def _prepare_batch(self, si: int, batch: dict) -> dict:
        """Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx]

        # DO NOT squeeze unless you're sure last dim = 1
        if cls.ndim > 1:
            cls = cls.squeeze(-1)

        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }
   
    

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

    model = YOLO(MODEL_CFG)
    model.model.load = False        # ← stop loading pretrained
    model.overrides['pretrained'] = False
    
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

    # #debug
    # # Dummy input: batch=1, 15 channels, 384x384
    # x = torch.randn(1, 15, 384, 384).to(device=device)

    # # Forward step by step
    # out = x
    # def iterate_yolo_layers(model_module):
    #     """递归遍历YOLO模型的所有层（处理嵌套Sequential/List）"""
    #     layers = []
    #     if isinstance(model_module, (torch.nn.Sequential, list, tuple)):
    #         for sub_module in model_module:
    #             layers.extend(iterate_yolo_layers(sub_module))
    #     elif isinstance(model_module, torch.nn.Module) and not isinstance(model_module, type(model.model)):
    #         # 排除DetectionModel本身，只保留实际的网络层
    #         layers.append(model_module)
    #     return layers

    # # 4. 获取所有可执行的层并逐层前向
    # all_layers = iterate_yolo_layers(model.model.model)  # 核心：拆解嵌套层
    # tem_list = []
    # for i, layer in enumerate(all_layers):
    #     try:
    #         # 禁用梯度计算（加速+避免显存占用）
    #         with torch.no_grad():
    #             # 打印层信息+输出形状
    #             layer_name = layer.__class__.__name__
    #             if layer_name == "Detect":
    #                 out = layer([tem_list[1],tem_list[2],tem_list[3]])
    #                 print(f"Layer {i:2d} | {layer_name:<20} | Output is a list with length {len(out)}")
    #                 # print("predict:",out)
    #             else:
    #                 out = layer(out)
    #                 print(f"Layer {i:2d} | {layer_name:<20} | Output shape: {tuple(out.shape)}")
    #             tem_list.append(out)
            
            
            
    #     except Exception as e:
    #         # 处理特殊层（如Detect层需要特定输入格式）
    #         print(f"Layer {i:2d} | {layer.__class__.__name__:<20} | Error: {str(e)}")
    #         break  # 遇到错误时停止遍历（可选）
    # print("temp_list[0]:",tem_list[0].shape)
    # print("temp_list[1]:",tem_list[1].shape)
    # print("temp_list[2]:",tem_list[2].shape)
    # print("temp_list[3]:",tem_list[3].shape)
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


   