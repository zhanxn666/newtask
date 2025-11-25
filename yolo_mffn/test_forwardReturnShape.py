
import torch

from module.MFFN_YOLO_Backbone import MFFN_YOLO_Backbone
B, C, H, W = 2, 15, 384, 384   # batch size 2, 5 views concatenated
dummy = torch.randn(B, C, H, W)

model = MFFN_YOLO_Backbone()
with torch.no_grad():
    out = model(dummy)
    print("Forward output shape:", out.shape)
#answer:
#[MFFN_YOLO_Backbone] MFFN output feature shape: (2, 64, 384, 384)
#Forward output shape: torch.Size([2, 64, 384, 384])