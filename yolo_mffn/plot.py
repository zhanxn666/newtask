import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_path = '/home/e706/zhanxiangning/newtask/yolo_mffn/runs/mffn_yolo/powerdata/results.csv'
df = pd.read_csv(csv_path)

# Convert epoch column to numpy
epochs = df['epoch'].to_numpy()

# 1. Plot losses
plt.figure(figsize=(12, 6))
plt.plot(epochs, df['train/box_loss'].to_numpy(), label='Train Box Loss')
plt.plot(epochs, df['train/cls_loss'].to_numpy(), label='Train Class Loss')
plt.plot(epochs, df['train/dfl_loss'].to_numpy(), label='Train DFL Loss')
plt.plot(epochs, df['val/box_loss'].to_numpy(), '--', label='Val Box Loss')
plt.plot(epochs, df['val/cls_loss'].to_numpy(), '--', label='Val Class Loss')
plt.plot(epochs, df['val/dfl_loss'].to_numpy(), '--', label='Val DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('YOLO Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/e706/zhanxiangning/newtask/yolo_mffn/runs/mffn_yolo/powerdata/losses.png')

# 2. Plot metrics: Precision, Recall, mAP50, mAP50-95 (for B class)
plt.figure(figsize=(12, 6))
plt.plot(epochs, df['metrics/precision(B)'].to_numpy(), label='Precision (B)')
plt.plot(epochs, df['metrics/recall(B)'].to_numpy(), label='Recall (B)')
plt.plot(epochs, df['metrics/mAP50(B)'].to_numpy(), label='mAP50 (B)')
plt.plot(epochs, df['metrics/mAP50-95(B)'].to_numpy(), label='mAP50-95 (B)')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('YOLO Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/e706/zhanxiangning/newtask/yolo_mffn/runs/mffn_yolo/powerdata/metrics.png')


# 3. Plot learning rates
plt.figure(figsize=(12, 6))
plt.plot(epochs, df['lr/pg0'].to_numpy(), label='LR pg0')
plt.plot(epochs, df['lr/pg1'].to_numpy(), label='LR pg1')
plt.plot(epochs, df['lr/pg2'].to_numpy(), label='LR pg2')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rates over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/e706/zhanxiangning/newtask/yolo_mffn/runs/mffn_yolo/powerdata/learning_rates.png')

