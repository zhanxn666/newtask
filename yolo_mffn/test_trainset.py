from ultralytics import YOLO
import torch
import numpy as np
import os
from pathlib import Path
import time
import cv2
# ====================== 1. 配置参数（替换为你的路径） ======================
MODEL_PATH = "/home/e706/zhanxiangning/newtask/yolo_mffn/runs/mffn_yolo/publicallpower9/weights/last.pt"
ROOT_IMGS = "/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower/images"
TRAIN_LABEL_DIR = "/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower/labels/train"
IMGSZ = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.1
TEST_SAMPLE_NUM = 10
MAX_TRAVERSE_NUM = 200  # 限制最大遍历数，避免卡死

# ====================== 2. 快速筛选有标签的样本 ======================
def fast_get_labeled_samples(label_dir, img_dir, suffix=".jpg", max_traverse=MAX_TRAVERSE_NUM):
    """仅读取标签文件，快速筛选有标签的样本名"""
    labeled_samples = []
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    
    for i, label_file in enumerate(label_files):
        if i >= max_traverse:
            break
        label_path = Path(label_dir) / label_file
        try:
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            # 有有效标签行
            if len(lines) > 0:
                img_name = label_file.replace(".txt", suffix)
                img_path = Path(img_dir) / img_name
                if img_path.exists():
                    labeled_samples.append(img_name)
                    if len(labeled_samples) >= TEST_SAMPLE_NUM:
                        break
        except Exception as e:
            continue
    return labeled_samples

# 执行快速筛选
train_img_dir = Path(ROOT_IMGS) / "train"
labeled_samples = fast_get_labeled_samples(TRAIN_LABEL_DIR, train_img_dir)

if len(labeled_samples) == 0:
    print("❌ 未找到有标签的样本！检查标签目录")
    exit()
print(f"快速找到 {len(labeled_samples)} 个有标签的样本: {labeled_samples[:5]}...")

# ====================== 3. 精准匹配数据集索引（基于你的 Dataset 源码） ======================
def get_sample_indices_by_path(dataset, sample_names):
    """遍历所有数据集样本，精准匹配有标签样本的索引（无数量限制）"""
    valid_indices = []
    # 遍历数据集全部样本（删除MAX_TRAVERSE_NUM限制）
    for idx in range(len(dataset.total_image_paths)):
        try:
            full_path = dataset.total_image_paths[idx]
            file_name = Path(full_path).name
            # 精准匹配文件名（包含空格/括号等特殊字符）
            if file_name in sample_names:
                valid_indices.append(idx)
                # 找到10个后停止
                if len(valid_indices) >= TEST_SAMPLE_NUM:
                    break
        except Exception as e:
            continue
    return valid_indices

# ====================== 4. 导入并初始化数据集 ======================
from dataset.mffn_yolo_dataset import MFFN_YOLO_Dataset

# 初始化数据集（完全复用你的方式）
data_cfg = {"train": "train"}
train_dataset = MFFN_YOLO_Dataset(
    root=[(ROOT_IMGS, {"image": {"path": data_cfg["train"], "suffix": ".jpg"}})],
    shape={"h": IMGSZ, "w": IMGSZ},
    label_dir=TRAIN_LABEL_DIR,
)

print(f"训练集样本总数: {len(train_dataset)}")

# ====================== 5. 匹配样本索引 ======================
valid_indices = get_sample_indices_by_path(train_dataset, labeled_samples)

if len(valid_indices) == 0:
    print(f"⚠️  未找到精准匹配的索引，尝试模糊匹配...")
    # 兜底：模糊匹配（兼容文件名大小写/后缀差异）
    valid_indices = []
    sample_stems = [Path(name).stem for name in labeled_samples]
    for idx in range(min(MAX_TRAVERSE_NUM, len(train_dataset.total_image_paths))):
        full_path = train_dataset.total_image_paths[idx]
        file_stem = Path(full_path).stem
        if any(stem in file_stem for stem in sample_stems):
            valid_indices.append(idx)
            if len(valid_indices) >= TEST_SAMPLE_NUM:
                break

if len(valid_indices) == 0:
    print("❌ 模糊匹配也未找到索引！使用前10个有标签样本手动加载")
    # 终极兜底：直接取前10个有标签样本名
    valid_samples = labeled_samples[:TEST_SAMPLE_NUM]
else:
    print(f"找到 {len(valid_indices)} 个匹配的样本索引: {valid_indices}")

# ====================== 6. 加载模型 ======================
print("\n加载模型中...")
model = YOLO(MODEL_PATH)
model = model.to(DEVICE)
model.eval()

# ====================== 7. 测试样本 ======================
has_pred_boxes = 0
total_gt_boxes = 0
total_pred_boxes = 0

# 情况1：找到匹配索引 → 用数据集加载
if len(valid_indices) > 0:
    for idx, sample_idx in enumerate(valid_indices):
        try:
            start_time = time.time()
            # 加载样本（你的Dataset返回格式）
            sample = train_dataset[sample_idx]
            img = sample["img"]  # 15通道张量 (15, 384, 384)
            instances = sample["instances"]
            gt_boxes = instances["bboxes"]  # (N,4) 归一化xywh
            gt_labels = instances["class"]  # (N,) 类别ID
            gt_box_num = len(gt_boxes) if gt_boxes.numel() > 0 else 0
            total_gt_boxes += gt_box_num

            # 打印样本信息
            print(f"\n===== 训练集样本 {idx+1}（原索引{sample_idx}） =====")
            print(f"加载耗时: {time.time()-start_time:.2f}s")
            print(f"图像张量形状: {img.shape}")  # 验证是15通道
            print(f"真实框数量: {gt_box_num}")
            print(f"输入数据范围: [{img.min():.4f}, {img.max():.4f}]")  # 验证归一化

            # 模型预测（禁用梯度）
            with torch.no_grad():
                img_input = img.unsqueeze(0).float().to(DEVICE)
                results = model.predict(
                    source=img_input,
                    conf=CONF_THRESH,
                    iou=0.5,
                    verbose=False,
                    device=DEVICE
                )

            # 解析预测结果
            pred_boxes = results[0].boxes
            pred_box_num = len(pred_boxes) if pred_boxes is not None else 0
            total_pred_boxes += pred_box_num

            if pred_box_num > 0:
                has_pred_boxes += 1
                pred_conf = pred_boxes.conf.cpu().numpy()
                pred_cls = pred_boxes.cls.cpu().numpy()
                print(f"预测框数量: {pred_box_num}")
                print(f"预测置信度: {pred_conf[:2]}")
                print(f"预测类别ID: {pred_cls[:2]}")
            else:
                print(f"预测框数量: 0")

        except Exception as e:
            print(f"样本 {sample_idx} 处理失败: {str(e)}")
            continue

# 情况2：未找到索引 → 手动加载（兜底）
else:
    print("\n手动加载有标签样本...")
    for idx, sample_name in enumerate(valid_samples):
        try:
            # 手动构建路径
            img_path = train_img_dir / sample_name
            label_path = Path(TRAIN_LABEL_DIR) / sample_name.replace(".jpg", ".txt")

            # 手动加载标签
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            gt_box_num = len(lines)
            total_gt_boxes += gt_box_num

            # 手动加载图像（复用数据集的逻辑）
            from MFFN.MFFN_COD_main.utils.io.image import read_color_array
            img = read_color_array(str(img_path))  # HWC uint8
            
            # 模拟数据集的增强和5视角生成（简化版）
            # 1. 归一化
            img = img / 255.0
            # 2. 调整尺寸到384×384
            img_resized = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
            # 3. 生成5视角并拼接为15通道
            c1 = cv2.resize(img, (IMGSZ*2, IMGSZ*2), interpolation=cv2.INTER_LINEAR)[:IMGSZ, :IMGSZ]
            o = img_resized
            c2 = cv2.resize(img, (int(IMGSZ*1.8), int(IMGSZ*1.8)), interpolation=cv2.INTER_LINEAR)[:IMGSZ, :IMGSZ]
            a1 = cv2.flip(img_resized, 0)
            a2 = cv2.flip(img_resized, -1)
            
            # 转换为张量并拼接
            c1 = torch.from_numpy(c1).permute(2, 0, 1).float()
            o = torch.from_numpy(o).permute(2, 0, 1).float()
            c2 = torch.from_numpy(c2).permute(2, 0, 1).float()
            a1 = torch.from_numpy(a1).permute(2, 0, 1).float()
            a2 = torch.from_numpy(a2).permute(2, 0, 1).float()
            img = torch.cat([c1, o, c2, a1, a2], dim=0)  # (15, 384, 384)

            print(f"\n===== 手动加载样本 {idx+1}（{sample_name}） =====")
            print(f"真实框数量: {gt_box_num}")
            print(f"图像张量形状: {img.shape}")
            print(f"输入数据范围: [{img.min():.4f}, {img.max():.4f}]")

            # 模型预测
            with torch.no_grad():
                img_input = img.unsqueeze(0).float().to(DEVICE)
                results = model.predict(
                    source=img_input,
                    conf=CONF_THRESH,
                    iou=0.5,
                    verbose=False,
                    device=DEVICE
                )

            # 解析结果
            pred_boxes = results[0].boxes
            pred_box_num = len(pred_boxes) if pred_boxes is not None else 0
            total_pred_boxes += pred_box_num

            if pred_box_num > 0:
                has_pred_boxes += 1
                pred_conf = pred_boxes.conf.cpu().numpy()
                pred_cls = pred_boxes.cls.cpu().numpy()
                print(f"预测框数量: {pred_box_num}")
                print(f"预测置信度: {pred_conf[:2]}")
                print(f"预测类别ID: {pred_cls[:2]}")
            else:
                print(f"预测框数量: 0")

        except Exception as e:
            print(f"样本 {sample_name} 处理失败: {str(e)}")
            continue

# ====================== 8. 测试总结 ======================
print("\n" + "="*50)
print("===== 测试总结 =====")
print(f"测试样本数: {TEST_SAMPLE_NUM}")
print(f"有真实框的样本数: {total_gt_boxes > 0}")
print(f"有预测框的样本数: {has_pred_boxes}")
print(f"总真实框数: {total_gt_boxes}")
print(f"总预测框数: {total_pred_boxes}")

if total_pred_boxes > 0:
    print("\n✅ 模型已学到特征！指标全0是计算逻辑问题")
    print("建议排查：")
    print("  1. 验证集标签格式是否与训练集一致")
    print("  2. YOLO配置中nc是否=6")
    print("  3. Detect头锚点是否匹配目标尺寸")
else:
    print("\n❌ 模型未学到有效特征！需修复以下问题：")
    print("  1. 检查Backbone是否返回dummy特征")
    print("  2. 确认15通道输入是否被正确处理")
    print("  3. 延长训练轮次（至少100轮）")
print("="*50)