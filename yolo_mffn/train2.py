from torch import nn
import torch
import yaml
from argparse import Namespace
from module.MFFN_YOLO_Backbone import MFFN_YOLO_Backbone
from ultralytics import YOLO
from dataset.mffn_yolo_dataset import MFFN_YOLO_Dataset
from utils.collate import mffn_yolo_collate_fn
import os
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
import ultralytics.nn.tasks as u_tasks

# ----------------------------------------------------------------------
# 1. 注册自定义模块（定义性代码，可放在全局）
# ----------------------------------------------------------------------
u_tasks.MFFN_YOLO_Backbone = MFFN_YOLO_Backbone  # 注册自定义Backbone

# ----------------------------------------------------------------------
# 2. 自定义Trainer类（定义性代码，可放在全局）
# ----------------------------------------------------------------------
class MyTrainer(DetectionTrainer):
    def __init__(self, train_dataset, val_dataset, custom_model, cfg, overrides=None, _callbacks=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Ensure overrides is always a dictionary
        if overrides is None:
            overrides = {}
        # 强制指定输入图像尺寸为384（与数据集输出一致）
        overrides['imgsz'] = 384
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
                batch_size=2,  # 减小batch_size（15通道+MFFN参数量大，避免OOM）
                shuffle=kwargs.get('shuffle', True),
                num_workers=4,  # 减少workers，避免spawn模式下资源竞争
                collate_fn=mffn_yolo_collate_fn,
                pin_memory=kwargs.get('pin_memory', True),
                multiprocessing_context='spawn',
            )
        else:  # validation
            return InfiniteDataLoader(
                self.val_dataset,
                batch_size=2,
                shuffle=kwargs.get('shuffle', False),
                num_workers=4,
                collate_fn=mffn_yolo_collate_fn,
                pin_memory=kwargs.get('pin_memory', True),
                multiprocessing_context='spawn',
            )

# ----------------------------------------------------------------------
# 3. 主执行逻辑（必须放在if __name__ == '__main__':中）
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------
    # 加载数据配置
    # -----------------------------
    DATA_YAML = "/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower.yaml"
    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    ROOT_IMGS = data_cfg["path"]                     # /home/.../powerdata/images
    TRAIN_IMG_DIR = f"{ROOT_IMGS}/{data_cfg['train']}"
    VAL_IMG_DIR = f"{ROOT_IMGS}/{data_cfg['val']}"
    TRAIN_LABEL_DIR = TRAIN_IMG_DIR.replace("images", "labels")
    VAL_LABEL_DIR = VAL_IMG_DIR.replace("images", "labels")
    IMGSZ = 384

    # -----------------------------
    # 构建数据集（执行性代码，放在主模块中）
    # -----------------------------
    train_dataset = MFFN_YOLO_Dataset(
        root=[(ROOT_IMGS, {"image": {"path": data_cfg["train"], "suffix": ".jpg"}})],
        shape={"h": IMGSZ, "w": IMGSZ},
        label_dir=TRAIN_LABEL_DIR,
    )
    print(f"✅ 训练数据集加载完成：{train_dataset}")

    val_dataset = MFFN_YOLO_Dataset(
        root=[(ROOT_IMGS, {"image": {"path": data_cfg["val"], "suffix": ".jpg"}})],
        shape={"h": IMGSZ, "w": IMGSZ},
        label_dir=VAL_LABEL_DIR,
    )
    print(f"✅ 验证数据集加载完成：{val_dataset}")

    # -----------------------------
    # 加载自定义模型（执行性代码，放在主模块中）
    # -----------------------------
    CFG_DIR = os.path.dirname(__file__)
    MODEL_CFG = os.path.join(CFG_DIR, "cfg", "yolo11_mffn2.yaml")  # 你的自定义YAML
    model = YOLO(MODEL_CFG)
    model.model.load = False  # 不加载预训练权重
    model.overrides['pretrained'] = False

    # 验证模型是否加载成功
    print("✅ 自定义模型加载完成：")
    print(f"模型结构：{model.model}")
    for name, module in model.model.named_modules():
        if isinstance(module, MFFN_YOLO_Backbone):
            print(f"✅ 成功加载自定义Backbone：{name}")

    # -----------------------------
    # 设备配置
    # -----------------------------
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"✅ 训练设备：{torch.cuda.get_device_name(device) if device != 'cpu' else 'CPU'}")

    # -----------------------------
    # 初始化Trainer并开始训练
    # -----------------------------
    trainer = MyTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        custom_model=model,  # 传入自定义模型
        cfg=MODEL_CFG,  # 使用自定义模型YAML（而非单独的cfg.yaml）
        overrides={
            "epochs": 10,  # 测试阶段先设10轮
            "device": device,
            "project": "runs/mffn_yolo",
            "name": "publicallpower_final",
            "amp": False,  # 禁用混合精度，稳定测试
            "plots": True,  # 开启日志绘图
            "batch": 2,  # 与dataloader的batch_size一致
        },
    )

    # 开始训练
    print("🚀 开始训练...")
    trainer.train()