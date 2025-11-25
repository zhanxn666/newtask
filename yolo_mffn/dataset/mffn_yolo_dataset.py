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
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],check_each_transform=False),#关闭检查
        )
    

    # ------------------------------------------------------------------
    # __getitem__ – 完全不碰 mask
    # ------------------------------------------------------------------
    
    def __getitem__(self, idx):
        # preemptively does what Albumentations refuses to do (fix tiny floating-point overflows)
        def clip_boxes(bboxes, format='yolo',eps=1e-6):
            # fix 1D bbox: convert (4,) → (1,4)
            if bboxes.ndim == 1:
                if bboxes.size == 4:
                    bboxes = bboxes.reshape(1, 4)
                elif bboxes.size == 5:
                    bboxes = bboxes.reshape(1, 5)
                else:
                    return []

            if bboxes.shape[0] == 0:
                return []
            if format == 'yolo':  # [cx, cy, w, h, class]
                bboxes = np.array(bboxes)
            # convert to [x_min, y_min, x_max, y_max]
            bboxes_xyxy = np.stack([
                bboxes[:, 0] - bboxes[:, 2] / 2,
                bboxes[:, 1] - bboxes[:, 3] / 2,
                bboxes[:, 0] + bboxes[:, 2] / 2,
                bboxes[:, 1] + bboxes[:, 3] / 2,
            ], axis=1)
        
            # clip to [0, 1]
            bboxes_xyxy = np.clip(bboxes_xyxy, 0.0, 1.0-eps)
        
            # back to yolo format
            bboxes[:, 0] = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2
            bboxes[:, 1] = (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2
            bboxes[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
            bboxes[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
            
            return bboxes.astype(np.float32)
            
        # 1. Read original image
        img_path = self.total_image_paths[idx]
        img = read_color_array(img_path)                      # HWC uint8
        H, W = img.shape[:2]

        # 2. Load YOLO labels
        raw_boxes  = self.boxes_list[idx]                     # (N,4) normalized xywh
        raw_labels = self.labels_list[idx]
        # super crazy clip or albumentations eat shit
        bboxes = clip_boxes(raw_boxes)

        # =======================================================
        # 强力 DEBUG：无论如何都打印 bbox 是否越界
        # =======================================================
        try:
            transformed = self.joint_trans(
            image=img,
            bboxes=bboxes,
            class_labels=raw_labels,
        )
        except Exception as e:
            print("\n\n❌❌ Albumentations ERROR at idx:", idx)
            print("Image path:", img_path)
            print("Label path:", self.label_paths[idx])
            print("Raw bboxes:", raw_boxes)
            print("Original bboxes:", bboxes)
            print("Image shape:", img.shape)
            print("Exception:", e)
            print(">>> Now re-raising...\n\n")
            raise e

        # 3. Joint augmentation
        # transformed = self.joint_trans(
        #     image=img,
        #     bboxes=bboxes,
        #     class_labels=raw_labels,
        # )

        img_aug   = transformed["image"]

        boxes_aug = np.array(transformed["bboxes"], dtype=np.float32) \
                        if transformed["bboxes"] else np.zeros((0, 4), np.float32)
        labels_aug = np.array(transformed["class_labels"], dtype=np.int64) \
                        if transformed["class_labels"] else np.zeros((0,), np.int64)


        # boxes_aug = np.array(transformed["bboxes"], dtype=np.float32) \
        #         if transformed["bboxes"] else np.zeros((0, 4), np.float32)
        # labels_aug = np.array(transformed["class_labels"], dtype=np.int64)

        # 4. Multi-view images (5 views)
        mv_imgs = ms_resize(img_aug, scales=(2.0, 1.0, 1.8),
                        base_h=self.base_h, base_w=self.base_w)

        flip_h = cv2.flip(img_aug, 0)
        flip_v = cv2.flip(img_aug, -1)

        a1 = ss_resize(flip_h, scale=1.0, base_h=self.base_h, base_w=self.base_w)
        a2 = ss_resize(flip_v, scale=1.0, base_h=self.base_h, base_w=self.base_w)

       
        
        H_raw, W_raw = img_aug.shape[:2]  # 增强后未缩放的图像尺寸
        target_h, target_w = self.base_h, self.base_w  # 目标尺寸：384×384
        def force_resize(img, target_h, target_w):
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        c1_resized = force_resize(mv_imgs[0], target_h, target_w)
        o_resized = force_resize(mv_imgs[1], target_h, target_w)
        c2_resized = force_resize(mv_imgs[2], target_h, target_w)
        a1_resized = force_resize(flip_h, target_h, target_w)
        a2_resized = force_resize(flip_v, target_h, target_w)
        # 6. Multi-view tensor dict → goes into "img"
        # multi_view = {
        #     "image_c1": torch.from_numpy(mv_imgs[0]).permute(2, 0, 1).float(),
        #     "image_o":  torch.from_numpy(mv_imgs[1]).permute(2, 0, 1).float(),
        #     "image_c2": torch.from_numpy(mv_imgs[2]).permute(2, 0, 1).float(),
        #     "image_a1": torch.from_numpy(a1).permute(2, 0, 1).float(),
        #     "image_a2": torch.from_numpy(a2).permute(2, 0, 1).float(),
        # }
        c1 = torch.from_numpy(c1_resized).permute(2, 0, 1).float()  # (3, 384, 384)
        o = torch.from_numpy(o_resized).permute(2, 0, 1).float()   # (3, 384, 384)
        c2 = torch.from_numpy(c2_resized).permute(2, 0, 1).float()  # (3, 384, 384)
        a1 = torch.from_numpy(a1_resized).permute(2, 0, 1).float()          # (3, 384, 384)
        a2 = torch.from_numpy(a2_resized).permute(2, 0, 1).float()          # (3, 384, 384)
        
        def resize_bboxes_for_force_resize(bboxes, H_raw, W_raw, target_h, target_w):
            """
            针对图像强制缩放，调整Bbox坐标：
            input: bboxes → (N,4) YOLO格式（归一化xywh，基于H_raw/W_raw）
            output: resized_bboxes → (N,4) 归一化xywh（基于target_h/target_w）
            """
            if len(bboxes) == 0:
                return bboxes
        
            # 计算图像实际缩放比例（强制缩放后的比例）
            scale_w = target_w / W_raw  # 宽度缩放比例
            scale_h = target_h / H_raw  # 高度缩放比例

            # 复制Bbox避免修改原数据
            resized_bboxes = bboxes.copy()

            # YOLO格式：(x, y, w, h) → 均为归一化值（0~1）
            # 步骤1：将归一化坐标转换为像素坐标（基于原始图像尺寸）
            resized_bboxes[:, 0] *= W_raw  # x（像素）= 归一化x × W_raw
            resized_bboxes[:, 1] *= H_raw  # y（像素）= 归一化y × H_raw
            resized_bboxes[:, 2] *= W_raw  # w（像素）= 归一化w × W_raw
            resized_bboxes[:, 3] *= H_raw  # h（像素）= 归一化h × H_raw

            # 步骤2：按图像缩放比例调整像素坐标
            resized_bboxes[:, 0] *= scale_w  # x → 按宽度比例缩放
            resized_bboxes[:, 1] *= scale_h  # y → 按高度比例缩放
            resized_bboxes[:, 2] *= scale_w  # w → 按宽度比例缩放
            resized_bboxes[:, 3] *= scale_h  # h → 按高度比例缩放

            # 步骤3：转换回归一化坐标（基于目标图像尺寸）
            resized_bboxes[:, 0] /= target_w  # 归一化x = 像素x / target_w
            resized_bboxes[:, 1] /= target_h  # 归一化y = 像素y / target_h
            resized_bboxes[:, 2] /= target_w  # 归一化w = 像素w / target_w
            resized_bboxes[:, 3] /= target_h  # 归一化h = 像素h / target_h

            # 步骤4：裁剪到[0,1]（避免缩放后溢出）
            resized_bboxes = np.clip(resized_bboxes, 0.0, 1.0)
            return resized_bboxes

        # 对增强后的Bbox执行同步缩放（仅主视角o的Bbox用于训练，其他视角无需）
        boxes_aug_resized = resize_bboxes_for_force_resize(
        boxes_aug, H_raw, W_raw, target_h, target_w
        )

        # Ensure correct dtype/format
        target_boxes = torch.from_numpy(boxes_aug_resized).float()      # (N,4)
        target_labels = torch.from_numpy(labels_aug).long()        # (N,)


        img_15ch = torch.cat([c1, o, c2, a1, a2], dim=0)  # (15, 384, 384)
        # print(f"[MFFN_YOLO_Dataset] img_15ch shape: {tuple(img_15ch.shape)}")
        # # 打印缩放前后的Bbox示例（调试用）
        # print(f"原始Bbox（增强后）：{boxes_aug[:2]}")  # 前2个Bbox（若有）
        # print(f"缩放后Bbox（适配384×384）：{boxes_aug_resized[:2]}")
        # print(f"图像原始尺寸：({H_raw}, {W_raw}) → 目标尺寸：({target_h}, {target_w})")
        # print(f"缩放比例（h,w）：({target_h/H_raw:.2f}, {target_w/W_raw:.2f})")
        ratio_pad = (1.0, 0, 0)
        # make sure return correct shape
        if target_boxes.ndim == 1 and target_boxes.numel() == 4:
            target_boxes = target_boxes.unsqueeze(0)  # (4) -> (1, 4)
        elif target_boxes.ndim == 1 and target_boxes.numel() == 0:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
    
        if target_labels.ndim == 0:  # scalar
            target_labels = target_labels.unsqueeze(0)  # () -> (1,)
        elif target_labels.ndim == 1 and target_labels.numel() == 0:
            target_labels = torch.zeros((0,), dtype=torch.long)
        
       

        # 7. Return YOLO-compatible format
        return {
            "img": img_15ch,                                    # ⭐ dict OK
            "instances": {
                "class":  target_labels,                          # (N,)
                "bboxes": target_boxes,                           # (N,4) xywh norm
            },
            "im_file": img_path,
            "resized_shape": (self.base_h, self.base_w),
            "ori_shape": img.shape,      # HWC
            "ratio_pad": ratio_pad,
        }

    '''
    def __getitem__(self, idx):
        # print(">> Using mffn_yolo dataset:", idx)


        # 1. 读取原始图像
        img_path = self.total_image_paths[idx]
        img = read_color_array(img_path)                     # HWC uint8

        # 2. 读取 YOLO 标签
        raw_boxes  = self.boxes_list[idx]                    # (N,4) normalized
        # print("raw_boxes:",raw_boxes)
        # raw_labels = self.labels_list[idx].tolist()
        raw_labels = self.labels_list[idx]
        # bboxes = raw_boxes.tolist()
        bboxes = raw_boxes

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
    '''
    def __len__(self):
        return len(self.total_image_paths)