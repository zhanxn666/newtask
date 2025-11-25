# utils/collate.py
import torch


def mffn_yolo_collate_fn(batch):
    """
    batch: list of items from __getitem__
    Each item = 
    {
            "img": multi_view,                                    # ⭐ dict OK
            "instances": {
                "class":  target_labels,                          # (N,)
                "bboxes": target_boxes,                           # (N,4) xywh norm
            },
            "im_file": img_path,
            "resized_shape": (self.base_h, self.base_w),
        }

        "multi_view": {
            "image_c1": Tensor[3,H,W],
            "image_o":  Tensor[3,H,W],
            "image_c2": Tensor[3,H,W],
            "image_a1": Tensor[3,H,W],
            "image_a2": Tensor[3,H,W],
        }
    """

    # -------------------------
    # 1) STACK MULTI-VIEW IMAGES
    # -------------------------
    img_batch = torch.stack([item["img"] for item in batch], dim=0)
    # img_batch = {}
    # view_keys = batch[0]["img"].keys()

    # for key in view_keys:
    #     img_batch[key] = torch.stack(
    #         [item["img"][key] for item in batch], dim=0
    #     )   # -> [B,3,H,W]
    '''
    view_keys = ["image_c1", "image_o", "image_c2", "image_a1", "image_a2"]

    # Collect 5 views
    view_tensors = {k: torch.stack([item["img"][k] for item in batch]) 
                    for k in view_keys}

    # Concatenate into ONE image tensor YOLO expects
    img = torch.cat(
        [view_tensors[k] for k in view_keys],
        dim=1  # channel dimension
    )  # shape: [B, 15, H, W]
    img_batch = img
    '''
    # -------------------------
    # 2) CONCAT TARGETS
    # -------------------------
    all_cls = []
    all_bboxes = []
    all_batch_idx = []

    for i, item in enumerate(batch):
        inst = item["instances"]
        num = len(inst["class"])

        if num > 0:
            all_cls.append(inst["class"])
            all_bboxes.append(inst["bboxes"])
            all_batch_idx.append(torch.full((num,), i, dtype=torch.int64))

    # Safe concat, even if no targets exist
    if len(all_cls) > 0:
        cls = torch.cat(all_cls, dim=0)
        bboxes = torch.cat(all_bboxes, dim=0)
        batch_idx = torch.cat(all_batch_idx, dim=0)
    else:
        cls = torch.zeros((0,), dtype=torch.int64)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.int64)

    # -------------------------
    # 3) META INFO
    # -------------------------
    im_files = [item["im_file"] for item in batch]
    resized_shapes = [item["resized_shape"] for item in batch]
    ori_shape = [item["ori_shape"] for item in batch]
    ratio_pad = [item["ratio_pad"] for item in batch]

    # guarantee cls shape (N,)
    if cls.ndim == 0:
        cls = cls.unsqueeze(0)

    # guarantee bboxes shape (N,4)
    if bboxes.ndim == 1:
        bboxes = bboxes.unsqueeze(0)
    # debug to check cls tensor shape
    # result:shape is rght im here
    try:
            for item in batch:
                a = item["instances"]["class"].shape[0]
            
    except Exception as e:
            print("batch[cls].shape[0] is wrong")
            print("batch[cls:]",batch[cls])
            print(e)
            raise e
    # -------------------------
    # 4) RETURN EXACT FORMAT YOU WANT
    # -------------------------
    return {
        "img": img_batch,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": im_files,
        "resized_shape": resized_shapes,
        "ori_shape": ori_shape,
        "ratio_pad": ratio_pad
    }
     #print(">> Using MY collate function. BatchSize =", len(batch))
'''
test code
   
H=W=32
num_obj=5
batch = [{
        "img": {
            "image_c1": torch.tensor([3,H,W]),
            "image_o":  torch.tensor([3,H,W]),
            "image_c2": torch.tensor([3,H,W]),
            "image_a1": torch.tensor([3,H,W]),
            "image_a2": torch.tensor([3,H,W]),
        },
        "instances": {
            "class":  torch.tensor([num_obj]),
            "bboxes": torch.tensor([num_obj,4]),   # xywh normalized
        },
        "im_file": "xxx",
        "resized_shape": (H,W)
    }]
collated = mffn_yolo_collate_fn(batch)
print(collated)
'''
'''
    multi_view_batch = {}
    # batch["img"]
    for key in batch[0]["multi_view"]:
        multi_view_batch[key] = torch.stack([item["multi_view"][key] for item in batch])
    targets = [item["targets"] for item in batch]
    return multi_view_batch, targets
'''