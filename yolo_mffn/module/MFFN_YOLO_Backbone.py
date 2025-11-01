import torch
import torch.nn as nn
from MFFN.MFFN_COD_main.methods.MFFN.MFFN import MFFN
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from MFFN.MFFN_COD_main.methods.module.base_model import BasicModelClass
from MFFN.MFFN_COD_main.methods.module.conv_block import ConvBNReLU
from MFFN.MFFN_COD_main.utils.builder import MODELS
from MFFN.MFFN_COD_main.utils.ops import cus_sample
import tensorly as tl


class MFFN_YOLO_Backbone(nn.Module):
    def __init__(self,c1=3):
        super().__init__()
        self.mffn = MFFN() 
        # adapt channels
       
        self.reduce1 = nn.Conv2d(64, 256, 3, stride=2, padding=1)   # 192×192
        self.reduce2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # 96×96
        self.reduce3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1) # 48×48
       
        
    def forward(self, x):
        # TRAINING: Real 5-view dict
        if isinstance(x, dict):
            c1 = x["image_c1"]
            o  = x["image_o"]
            c2 = x["image_c2"]
            a1 = x["image_a1"]
            a2 = x["image_a2"]

            out = self.mffn.body(c1, o, c2, a1, a2)
            dec = out['final_features']

    
            p3 = self.reduce1(dec)
            p4 = self.reduce2(p3)
            p5 = self.reduce3(p4)
            dec_out = [p3, p4, p5]
            return dec_out

        # MODEL BUILD: Dummy 4D tensor → return fake YOLO features
        else:
            # MODEL BUILD: Return fake tensor with 3 channels
            B = x.shape[0]
            device = x.device
            return torch.zeros(B, 3, 48, 48, device=device)  # ← 3 channels!
