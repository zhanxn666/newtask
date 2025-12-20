import torch
import torch.nn as nn
from MFFN.MFFN_COD_main.methods.MFFN.MFFN import MFFN
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from MFFN.MFFN_COD_main.methods.module.base_model import BasicModelClass
from MFFN.MFFN_COD_main.methods.module.conv_block import ConvBNReLU
from MFFN.MFFN_COD_main.utils.builder import MODELS
from MFFN.MFFN_COD_main.utils.ops import cus_sample
import tensorly as tl
import torch.nn.functional as F

class MFFN_YOLO_Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mffn = MFFN() 
        self.conv = nn.Conv2d(
            in_channels=15, 
            out_channels=64, 
            kernel_size=3, 
            padding=1  # 3×3卷积+padding=1 → 输出尺寸=输入尺寸
        )
        # adapt channels
    def forward(self, x):
        
        # print("[FORWARD INPUT SHAPE]", x.shape)
        
        # if x.shape[1] != 15:
        # #     print(f"[MFFN_YOLO_Backbone] Dummy input detected, returning compatible features")
        # #     print("dummy tensor shape:",x.shape)
        # # Return features with the same spatial dimensions but correct channels
        # # that your MFFN normally outputs

        #     B = x.shape[0]
        #     device = x.device
        #     dtype = x.dtype
        #     # Return dummy features that match what MFFN would output
        #     return torch.zeros(B, 64, 384, 384, device=device, dtype=dtype)  # Match your MFFN output channels
        
        # print(">>>>>using main branch not dummy return tensor>>>>>")
        # TRAINING: Real 5-view dict
        if x.shape[1] == 3:  # stride-init dummy
                x = x.repeat(1, 5, 1, 1)  # 3 -> 15
        if x.shape[1] != 15:
            raise ValueError(f"Expected 15ch input, got {x.shape[1]}")
        c1 = x[:, 0:3, :, :]   # (B, 3, 384, 384) → 视角1
        o = x[:, 3:6, :, :]    # (B, 3, 384, 384) → 主视角
        c2 = x[:, 6:9, :, :]   # (B, 3, 384, 384) → 视角2
        a1 = x[:, 9:12, :, :]  # (B, 3, 384, 384) → 辅助视角1
        a2 = x[:, 12:15, :, :] # (B, 3, 384, 384) → 辅助视角2
        # print("x.shape:",x.shape)
        # print("c1 shape:",c1.shape)
        # print("o shape:",o.shape)
        # print("c2 shape:",c2.shape)
        # print("a1 shape:",a1.shape)
        # print("a2 shape:",a2.shape)
        out = self.mffn.body(c1, o, c2, a1, a2)
        dec = out['final_features']
        # print(f"[MFFN_YOLO_Backbone] MFFN output feature shape: {tuple(dec.shape)}")
        return dec
        '''
        #debug for MFFN body
        if x.shape[1] != 15:
            B = x.shape[0]
            device = x.device
            dtype = x.dtype
            return torch.zeros(B, 64, 384, 384, device=device, dtype=dtype)
        # print("input tensor shape:",x.shape)
        dec = self.conv(x)
        
        # print(f"[ModifiedBackbone] Conv output feature shape: {tuple(dec.shape)}")
        return dec
       '''
