# dataset/mffn_yolo_dataset.py
import os
import cv2
import torch
import numpy as np
import albumentations as A
from MFFN.MFFN_COD_main.dataset.transforms.rotate import UniRotate

from MFFN.MFFN_COD_main.dataset.MultiView_cod import MFFN_COD_TrainDataset
from MFFN.MFFN_COD_main.dataset.transforms.resize import (
    ms_resize, ss_resize
)
from MFFN.MFFN_COD_main.utils.io.image import read_color_array

# ----------------------------------------------------------------------
# Helper: resize boxes (identical to image resize)
# ----------------------------------------------------------------------
def _resize_boxes(boxes, src_h, src_w, dst_h, dst_w):
    boxes = boxes.copy()
    boxes[:, 0] *= dst_w / src_w   # x
    boxes[:, 1] *= dst_h / src_h   # y
    boxes[:, 2] *= dst_w / src_w   # w
    boxes[:, 3] *= dst_h / src_h   # h
    return boxes

def ms_resize_boxes(boxes, scales, base_h, base_w):
    return [_resize_boxes(boxes, base_h, base_w,
                         int(base_h * s), int(base_w * s)) for s in scales]

def ss_resize_boxes(boxes, scale, base_h, base_w):
    h, w = int(base_h * scale), int(base_w * scale)
    return _resize_boxes(boxes, base_h, base_w, h, w)


# ----------------------------------------------------------------------
# YOLO Dataset – NO MASK, FULLY COMPATIBLE WITH MFFN COD utils
# ----------------------------------------------------------------------
class MFFN_YOLO_Dataset(MFFN_COD_TrainDataset):
    """
    - 5-view generation (c1, o, c2, a1, a2)  ← 完全保留
    - YOLO bounding-boxes (xywh, normalized) ← 同步变换
    - **Zero mask loading** (dummy mask dir only for parent compatibility)
    """
    
    def __init__(self, root, shape, label_dir, img_suffix=".jpg", **kwargs):
        """
        root:  can be
               * str  → "/data/powerdata"
               * list → [("/data/powerdata", {"image": {"path": "train", "suffix": ".jpg"}})]
        shape: {"h": 384, "w": 384}
        label_dir: folder containing YOLO *.txt files
        img_suffix: ".jpg" or ".png"
        """
        # --------------------------------------------------------------
        # 1. 统一 root 格式
        # --------------------------------------------------------------
        DUMMY_MASK_DIR = "/home/e706/zhanxiangning/newtask/yolo_mffn/dummy_masks"
        os.makedirs(DUMMY_MASK_DIR, exist_ok=True)  # ← 关键：确保存在
        if isinstance(root, str):
            root = [(root, {})]
        elif isinstance(root, (list, tuple)) and isinstance(root[0], str):
            root = [(root[0], {})]

        # --------------------------------------------------------------
        # 2. 构造 **完全兼容 MFFN** 的 dummy_root
        # --------------------------------------------------------------
        dummy_root = []
        for base_dir, user_info in root:
        # ----- 提取用户提供的 image 子目录 -----
            img_sub = user_info.get("image", "")
            if isinstance(img_sub, dict):
                img_path = img_sub.get("path", "")
                img_suf  = img_sub.get("suffix", img_suffix)
            else:
                img_path = img_sub
                img_suf  = img_suffix

        # ----- 构造完整的 dataset_info（必须有 "root"） -----
            dataset_info = {
                "root": base_dir,                                   # ← 关键！
                "image": {"path": img_path, "suffix": img_suf},
                "mask":  {"path": DUMMY_MASK_DIR, "suffix": ".png"}, # ← 空文件夹
            }

            dummy_root.append(
                ("powerdata", dataset_info)  # dataset_name 可随意
            )
        # --------------------------------------------------------------
        # 3. 调用父类 → 自动生成 self.total_image_paths
        # --------------------------------------------------------------
        super().__init__(root=dummy_root, shape=shape, **kwargs)

        self.label_dir = label_dir
        self.base_h = shape["h"]
        self.base_w = shape["w"]
        self.img_suffix = img_suf

        # --------------------------------------------------------------
        # 4. YOLO label paths (aligned with total_image_paths)
        # --------------------------------------------------------------
        self.label_paths = [
            os.path.join(
                label_dir,
                os.path.splitext(os.path.basename(p))[0] + ".txt"
            )
            for p in self.total_image_paths
        ]

        # --------------------------------------------------------------
        # 5. Pre-load boxes / classes (faster __getitem__)
        # --------------------------------------------------------------
        self.boxes_list  = []
        self.labels_list = []
        for lp in self.label_paths:
            if os.path.exists(lp):
                boxes, labels = [], []
                with open(lp, "r") as f:
                    for line in f.readlines():
                        c, x, y, w, h = map(float, line.strip().split())
                        boxes.append([x, y, w, h])
                        labels.append(int(c))
                boxes  = np.array(boxes,  dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
            else:
                boxes  = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,),   dtype=np.int64)
            self.boxes_list.append(boxes)
            self.labels_list.append(labels)

        # --------------------------------------------------------------
        # 6. Augmentations (image + bboxes, **no mask**)
        # --------------------------------------------------------------
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    # ------------------------------------------------------------------
    # __getitem__ – 完全不碰 mask
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # 1. 读取原始图像
        img_path = self.total_image_paths[idx]
        img = read_color_array(img_path)                     # HWC uint8

        # 2. 读取 YOLO 标签
        raw_boxes  = self.boxes_list[idx]                    # (N,4) normalized
        raw_labels = self.labels_list[idx].tolist()
        bboxes = raw_boxes.tolist()

        # 3. 联合增强（图像 + bboxes）
        transformed = self.joint_trans(
            image=img,
            bboxes=bboxes,
            class_labels=raw_labels,
        )
        img_aug   = transformed["image"]
        boxes_aug = np.array(transformed["bboxes"], dtype=np.float32) \
                    if transformed["bboxes"] else np.zeros((0,4), np.float32)
        labels_aug = np.array(transformed["class_labels"], dtype=np.int64)

        # 4. 5-view 生成
        mv_imgs = ms_resize(img_aug, scales=(2.0, 1.0, 1.8),
                            base_h=self.base_h, base_w=self.base_w)

        flip_h = cv2.flip(img_aug, 0)
        flip_v = cv2.flip(img_aug, -1)
        a1 = ss_resize(flip_h, scale=1.0, base_h=self.base_h, base_w=self.base_w)
        a2 = ss_resize(flip_v, scale=1.0, base_h=self.base_h, base_w=self.base_w)

        # 5. 同步缩放 boxes
        mv_boxes = ms_resize_boxes(boxes_aug, scales=(2.0, 1.0, 1.8),
                                   base_h=self.base_h, base_w=self.base_w)
        a1_boxes = ss_resize_boxes(boxes_aug, scale=1.0,
                                   base_h=self.base_h, base_w=self.base_w)
        a2_boxes = ss_resize_boxes(boxes_aug, scale=1.0,
                                   base_h=self.base_h, base_w=self.base_w)

        # 6. 主目标（scale=1.0）—— YOLO head 只需要这一套
        target_boxes = ss_resize_boxes(boxes_aug, scale=1.0,
                                       base_h=self.base_h, base_w=self.base_w)

        # 7. 组装 multi_view dict（**无 mask**）
        multi_view = {
            "image_c1": torch.from_numpy(mv_imgs[0]).permute(2, 0, 1),
            "image_o":  torch.from_numpy(mv_imgs[1]).permute(2, 0, 1),
            "image_c2": torch.from_numpy(mv_imgs[2]).permute(2, 0, 1),
            "image_a1": torch.from_numpy(a1).permute(2, 0, 1),
            "image_a2": torch.from_numpy(a2).permute(2, 0, 1),
        }

        targets = {
            "boxes":  torch.from_numpy(target_boxes).float(),   # (N,4) xywh
            "labels": torch.from_numpy(labels_aug).long(),
            "img_id": torch.tensor([idx]),
        }

        return {"multi_view": multi_view, "targets": targets}

    def __len__(self):
        return len(self.total_image_paths)