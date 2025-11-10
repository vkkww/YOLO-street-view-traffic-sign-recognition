import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 防止MKL冲突

# 设置matplotlib支持中文显示的字体
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

from ultralytics import YOLO


def main():
    # 加载YOLO模型
    model = YOLO("yolov8n.pt")  # 如果不存在会自动下载预训练模型

    # 开始训练
    results = model.train(data="BJrdData/BJrd.yaml", epochs=150, imgsz=320,batch=8,workers=0,device=0)  # 注意：imgsz应为整数或单个数值


if __name__ == "__main__":
    main()