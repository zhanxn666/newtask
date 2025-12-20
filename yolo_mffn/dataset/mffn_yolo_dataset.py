# dataset/mffn_yolo_dataset.py
import os
import cv2
import torch
import numpy as np
import albumentations as A
import yaml
from MFFN.MFFN_COD_main.dataset.transforms.rotate import UniRotate

from MFFN.MFFN_COD_main.dataset.MultiView_cod import MFFN_COD_TestDataset, MFFN_COD_TrainDataset
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
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],check_each_transform=False),#关闭检查
        )
        

    # ------------------------------------------------------------------
    # __getitem__ – 完全不碰 mask
    # ------------------------------------------------------------------
    
    def __getitem__(self, idx):
        def clip_yolo_boxes(bboxes, eps=1e-8):
            """
            boxes: (N,4) YOLO xywh normalized
            """
            # 1. 处理空输入（返回空numpy数组，而非列表）
            if isinstance(bboxes, list):
                if not bboxes:
                    return np.empty((0, 4), dtype=np.float32)
                bboxes = np.array(bboxes, dtype=np.float32)
            elif not isinstance(bboxes, np.ndarray):
                raise TypeError(f"bboxes must be list or np.ndarray, got {type(bboxes)}")

            # 2. 处理一维数组（单个bbox）→ 二维数组 (1,4)
            if bboxes.ndim == 1:
                if bboxes.size == 4:
                    bboxes = bboxes.reshape(1, 4)
                else:
                    return np.empty((0, 4), dtype=np.float32)  # 维度错误返回空数组

            # 3. 过滤无效行（比如长度不是4的行）
            if bboxes.shape[1] != 4:
                bboxes = bboxes[:, :4]  # 只保留前4列（xywh）
                if bboxes.shape[1] != 4:
                    return np.empty((0, 4), dtype=np.float32)   
            # 4. YOLO xywh → xyxy（便于裁剪边界）
            bboxes_xyxy = np.stack([
                bboxes[:, 0] - bboxes[:, 2] / 2,  # x_min
                bboxes[:, 1] - bboxes[:, 3] / 2,  # y_min
                bboxes[:, 0] + bboxes[:, 2] / 2,  # x_max
                bboxes[:, 1] + bboxes[:, 3] / 2   # y_max
            ], axis=1)

            # 5. 强制裁剪到 [0.0, 1.0-eps]（彻底消除超出值，包括浮点精度）
            bboxes_xyxy = np.clip(bboxes_xyxy, 0.0, 1.0 - eps)

            # 6. 过滤无效框（x_min >= x_max 或 y_min >= y_max）
            valid = (bboxes_xyxy[:, 0] < bboxes_xyxy[:, 2]) & (bboxes_xyxy[:, 1] < bboxes_xyxy[:, 3])
            bboxes_xyxy = bboxes_xyxy[valid]

            # 7. xyxy → YOLO xywh（还原格式）
            bboxes_yolo = np.stack([
                (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2,  # cx
                (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2,  # cy
                bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0],        # w
                bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]         # h
            ], axis=1)

            return bboxes_yolo.astype(np.float32)

        # 1. Read original image
        img_path = self.total_image_paths[idx]
        img = read_color_array(img_path)   
        image0 = cv2.flip(img, 0, dst=None)  # 作水平镜像翻转
        image1 = cv2.flip(img, -1, dst=None)                   # HWC uint8

        # 2. Load YOLO labels
        raw_boxes  = self.boxes_list[idx]                     # (N,4) normalized xywh
        raw_labels = self.labels_list[idx]
        # super crazy clip for albumentations bug
        raw_boxes = clip_yolo_boxes(raw_boxes)

        # bboxes = clip_boxes(raw_boxes)
        bboxes = raw_boxes

        try:
            if (bboxes>1.0).any() or (bboxes<0.0).any():
                print("error bboxes", bboxes)
        except:
            print("bboxes error:", bboxes)
      
        
        transformed = self.joint_trans(
            image=img,
            bboxes=bboxes,
            class_labels=raw_labels,
            allow_out_of_bounds=True)
        
        transformed0 = self.joint_trans(
            image=image0,
             bboxes=bboxes,
            class_labels=raw_labels,
            allow_out_of_bounds=True)
        
        transformed1 = self.joint_trans(
            image=image1,
            bboxes=bboxes,
            class_labels=raw_labels,
            allow_out_of_bounds=True)
        


        img_aug   = transformed["image"]

        image0_aug = transformed0["image"]

        image1_aug = transformed1["image"]

        boxes_aug = np.array(transformed["bboxes"])
        boxes_aug = clip_yolo_boxes(boxes_aug)
        
        labels_aug = np.array(transformed["class_labels"])
        #this part is different from MFFN_COD_main, (2.0,1.0,1.8) -> (1.0,1.0,1.0)
        images = ms_resize(img_aug, scales=(1.0,1.0,1.0), base_h=self.base_h, base_w=self.base_w)
        image0 = ss_resize(image0_aug, scale=1.0, base_h=self.base_h, base_w=self.base_w)
        image1 = ss_resize(image1_aug, scale=1.0, base_h=self.base_h, base_w=self.base_w)

        image_c_1 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_o = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_c_2 = torch.from_numpy(images[2]).permute(2, 0, 1)
        image_a_1 = torch.from_numpy(image0).permute(2, 0, 1)
        image_a_2 = torch.from_numpy(image1).permute(2, 0, 1)
        # print("images shape", image_c_1.shape, image_o.shape, image_c_2.shape, image_a_1.shape, image_a_2.shape)
        #images shape torch.Size([3, 384, 384]) torch.Size([3, 384, 384]) torch.Size([3, 384, 384]) torch.Size([3, 384, 384]) torch.Size([3, 384, 384])
        # Ensure correct dtype/format
        target_boxes = torch.tensor(boxes_aug, dtype=torch.float32)      # (N,4)
        target_labels = torch.from_numpy(labels_aug).long()        # (N,)

        # print("target_boxes", target_boxes)
        # print("target_labels", target_labels)

        img_15ch = torch.cat([image_c_1, image_o, image_c_2, image_a_1, image_a_2], dim=0)  # (15, 384, 384)
        # 7. Return YOLO-compatible format
        # print(">>>>>>>>",self.base_h,self.base_w)
        return {
            "img": img_15ch,                                    # ⭐ dict OK
            "instances": {
                "class":  target_labels,                          # (N,)
                "bboxes": target_boxes,                           # (N,4) xywh norm
            },
            "im_file": img_path,
            "resized_shape": (self.base_h, self.base_w),
            "ori_shape": img.shape[:2],      # HWC
            "ratio_pad": (1.0, 1.0, 0.0, 0.0),
        }
    
    def __len__(self):
        return len(self.total_image_paths)
    
