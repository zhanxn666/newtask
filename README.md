# MFFN-YOLO

This repository contains an experimental implementation of a **MFFN: Multi-view Feature Fusion Network for Camouflaged Object Detection**
integrated into the **Ultralytics YOLO (v11) framework** to support **15-channel multi-view input**.

---

1. Model Architecture:


    Backbone: MFFN without last ConvBlock


    Head: YOLO detection head


3. Environment Setup:


  ```python
Pillow == 8.1.2
addict == 2.4.0
albumentations == 1.0.0
matplotlib == 3.4.2
numpy == 1.19.2
opencv_python_headless == 4.5.1.48
openpyxl == 3.0.7
pysodmetrics == 1.2.4
scipy == 1.6.2
timm == 0.4.12
tqdm == 4.59.0
ttach == 0.0.3
yapf == 0.31.0
ultralytics
 ```
4. Training & Validation:


run train.py to train the model, don't forget to change the dataset path(in /cfg/cfg.yaml) to your own dataset path and change nc to match your dataset in yolo11_mffn2.yaml. You can change the configurations in cfg.yaml.

6. Important note:


We have some small modifications in albumentations and ultralytics to make sure YOLO can accept 15 channels input, so make sure you have done these modifications before you run train.py.

(1)albumentations/augmentations/bbox_utils.py
  line 251 modified as:
  ```python
    try:
      check_bbox(bbox)
    except Exception as e:
      pass
 ```

      
(2) ultralytics/models/yolo/detect/val.py
line 140 modified as:
```python
  cls = batch["cls"][idx].flatten()
```


(3)ultralytics/nn/tasks.py
line 1628 add these:
```python
  elif m.__name__ = "MFFN_YOLO_Backbone":
    c2 = args[0]
```
