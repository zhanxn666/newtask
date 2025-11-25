import warnings
warnings.filterwarnings('ignore')
import os, tqdm
from ultralytics import YOLO
from module.MFFN_YOLO_Backbone import MFFN_YOLO_Backbone

# ## test simgle
# if __name__ == '__main__':
#     # 直接指定你要测试的YAML文件
#     yaml_file = 'yolo11_mffn.yaml'  # 替换为你要测试的具体文件名
#     yaml_path = f'/home/e706/zhanxiangning/newtask/yolo_mffn/cfg/{yaml_file}'
    
#     try:
#         print(f"Testing: {yaml_file}")
#         model = YOLO(yaml_path)
#         model.info(detailed=True)
#         model.profile([640, 640])
#         model.fuse()
#         print("✓ All tests passed!")
#     except Exception as e:
#         print(f"❌ Error: {e}")

from module.MFFN_YOLO_Backbone import MFFN_YOLO_Backbone
import torch

model = MFFN_YOLO_Backbone()

# 构造一个假的五视图输入
x = {
    "image_c1": torch.randn(1, 3, 384, 384),
    "image_o":  torch.randn(1, 3, 384, 384),
    "image_c2": torch.randn(1, 3, 384, 384),
    "image_a1": torch.randn(1, 3, 384, 384),
    "image_a2": torch.randn(1, 3, 384, 384),
}

out = model(x)

