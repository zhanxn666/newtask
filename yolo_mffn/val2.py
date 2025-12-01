# val_debug.py  ← 2025 working version
from types import MethodType
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.nms import non_max_suppression

# 1. Load your model
model = YOLO('/home/e706/zhanxiangning/newtask/yolo_mffn/runs/mffn_yolo/publicallpower9/weights/last.pt')

# Print the REAL class mapping stored inside the .pt file (this is the truth!)
print("Class names saved in the .pt file:")
print(model.model.names)
print("Number of classes:", model.model.nc)

# 2. Custom postprocess that prints raw predictions
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.nms import non_max_suppression



# ==========================
# 1. Define debug postprocess
# ==========================
def debug_postprocess(self, preds, img, orig_imgs):
    print("\n[DEBUG] RAW preds:", preds.shape)

    obj = preds[0, :, 4]
    topk = obj.topk(min(10, obj.numel())).indices
    print("[DEBUG] top-10 obj conf:", obj[topk].cpu().numpy())

    # run NMS
    outputs = non_max_suppression(
        preds,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300
    )

    print(f"[DEBUG] NMS kept {len(outputs[0])} boxes")
    if len(outputs[0]) > 0:
        print("[DEBUG] classes:", outputs[0][:, -1].cpu().numpy())
        print("[DEBUG] confs   :", outputs[0][:, 4].cpu().numpy())

    return outputs


# ==========================
# 2. Build validator MANUALLY
# ==========================
validator = DetectionValidator(model=model)


# ==========================
# 3. Patch validator.predictor.postprocess
# ==========================
# validator.get_predictor() CREATES the predictor used by validation
predictor = validator.get_predictor(model=model)

# monkey-patch the postprocess
predictor.postprocess = debug_postprocess.__get__(predictor)

print(">>> Patched validator predictor successfully.")


# ==========================
# 4. Run validation
# ==========================
validator(
    model=model,
    data='/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower.yaml',
    imgsz=384,
    split='val',     # or 'train'
    conf=0.001,
)

# result = model.val(
#                   # source='/home/lenovo/data/liujiaji/Datasets-Power/privatepower/pin/rust/img',
#                   source='/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower/images',
#                   split='train',
#                   imgsz=384,
#                   project='runs/detect',
#                   plots=True,
#                   name='fea',
#                   conf=0.001,
#                   # save=True,
#                 #   conf=0.3
#                 )
# 4. Run validation — you will see tons of debug output
# results = model.val(
#     data='/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower.yaml',  # correct path!
#     split='train',
#     imgsz=384,
#     batch=4,
#     conf=0.001,
#     iou=0.6,
#     plots=False,
#     verbose=False,
#     save=False,
#     workers=4
# )

# print("\nFinal results:")
# print("mAP50-95 :", results.box.map)
# print("mAP50    :", results.box.map50)