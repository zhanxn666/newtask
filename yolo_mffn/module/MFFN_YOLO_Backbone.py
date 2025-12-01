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
import torch.nn.functional as F

class MFFN_YOLO_Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mffn = MFFN() 
        # adapt channels
        '''
        self.reduce1 = nn.Conv2d(64, 256, 3, stride=2, padding=1)   # 192×192
        self.reduce2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # 96×96
        self.reduce3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1) # 48×48
        '''
        
    def forward(self, x):
        '''
        if x.shape[1] == 15:
            # split into 5 images (each 3 channels)
            c1 = x[:, 0:3]
            o  = x[:, 3:6]
            c2 = x[:, 6:9]
            a1 = x[:, 9:12]
            a2 = x[:, 12:15]
            out = self.mffn.body(c1, o, c2, a1, a2)
            dec = out['final_features']
            print(f"[MFFN_YOLO_Backbone] MFFN output feature shape: {tuple(dec.shape)}")
            return dec
        else:
            B = x.shape[0]
            device = x.device
            dtype = x.dtype

            return torch.zeros(B, 3, 384, 384, device=device, dtype=dtype)

        '''

        # print("[FORWARD INPUT SHAPE]", x.shape)

        if x.shape[1] != 15:
        #     print(f"[MFFN_YOLO_Backbone] Dummy input detected, returning compatible features")
        #     print("dummy tensor shape:",x.shape)
        # Return features with the same spatial dimensions but correct channels
        # that your MFFN normally outputs

            B = x.shape[0]
            device = x.device
            dtype = x.dtype
            # Return dummy features that match what MFFN would output
            return torch.zeros(B, 64, 384, 384, device=device, dtype=dtype)  # Match your MFFN output channels
        
        # print(">>>>>using main branch not dummy return tensor>>>>>")
        # TRAINING: Real 5-view dict
        # print("x.shape:",x.shape)
        # c1 = x["image_c1"]
        # o  = x["image_o"]
        # c2 = x["image_c2"]
        # a1 = x["image_a1"]
        # a2 = x["image_a2"]
        c1 = x[:, 0:3, :, :]   # (B, 3, 384, 384) → 视角1
        o = x[:, 3:6, :, :]    # (B, 3, 384, 384) → 主视角
        c2 = x[:, 6:9, :, :]   # (B, 3, 384, 384) → 视角2
        a1 = x[:, 9:12, :, :]  # (B, 3, 384, 384) → 辅助视角1
        a2 = x[:, 12:15, :, :] # (B, 3, 384, 384) → 辅助视角2
        # print(f"c1 shape: {tuple(c1.shape)}")
        # print(f"o shape: {tuple(o.shape)}")
        # print(f"c2 shape: {tuple(c2.shape)}")
        # print(f"a1 shape: {tuple(a1.shape)}")
        # print(f"a2 shape: {tuple(a2.shape)}")
        out = self.mffn.body(c1, o, c2, a1, a2)
        dec = out['final_features']
        # print(f"[MFFN_YOLO_Backbone] MFFN output feature shape: {tuple(dec.shape)}")
        return dec
        
        # else:
        #     B = x.shape[0]
        #     device = x.device
        #     dtype = x.dtype

        #     return torch.zeros(B, 3, 256, 256, device=device, dtype=dtype)
        '''
            p3 = self.reduce1(dec)
            p4 = self.reduce2(p3)
            p5 = self.reduce3(p4)
            dec_out = p3, p4, p5
            print(f"[MFFN_YOLO_Backbone] Output feature shapes:")
            for i, o in enumerate(dec_out):
                print(f"  P{i+3}: {tuple(o.shape)}")
        '''
        '''
        #this is the problematic part, when yolo uses original dataset loader, it will pass a tensor instead of dict, then 
        #it will go to this branch, so the model will output a fake tensor instead of real features which make model learn nothing
        # MODEL BUILD: Dummy 4D tensor → return fake YOLO features
        else:
            # MODEL BUILD: Return fake tensor with 3 channels
            
            B = x.shape[0]
            device = x.device
            return torch.zeros(B, 3, 48, 48, device=device)  # ← 3 channels!
            
            # Model build: fake output
            # B, _, H, W = x.shape
            # return torch.zeros(B, 64, H, W, device=x.device)
        '''
        '''
        
        '''
