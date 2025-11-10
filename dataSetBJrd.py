import os
import shutil
import random

# 定义路径
image_dir = r"G:\myStuff\study\3_2\spatial_data\StreetImg\ultralytics\BJrdData\dataTest2\images"
label_dir = r"G:\myStuff\study\3_2\spatial_data\StreetImg\ultralytics\BJrdData\dataTest2\labels"
target_base_dir = r"G:\myStuff\study\3_2\spatial_data\StreetImg\ultralytics\BJrdData\BJrdSet\test2"

os.makedirs(target_base_dir, exist_ok=True)

def collect_pairs(image_dir, label_dir):
    valid_pairs = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png'):
            base_name = os.path.splitext(image_file)[0]
            label_file = f"{base_name}.txt"

            # 验证标签文件是否存在
            if os.path.exists(os.path.join(label_dir, label_file)):
                valid_pairs.append((label_file, image_file))
            else:
                print(f"警告：缺失标签 {label_file}")
    return valid_pairs

def split_dataset(pairs, ratio=0.8):
    random.shuffle(pairs)
    split_index = int(len(pairs) * ratio)
    return pairs[:split_index], pairs[split_index:]

def process_pairs(pairs, target_dir):
    os.makedirs(os.path.join(target_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
    for label, image in pairs:
        shutil.copy(os.path.join(label_dir, label), os.path.join(target_dir, 'labels', label))
        shutil.copy(os.path.join(image_dir, image), os.path.join(target_dir, 'images', image))

# 收集所有有效的配对
valid_pairs = collect_pairs(image_dir, label_dir)

# 划分数据集
train_pairs, val_pairs = split_dataset(valid_pairs)

# 处理并分配到目标目录
process_pairs(train_pairs, os.path.join(target_base_dir, 'train'))
process_pairs(val_pairs, os.path.join(target_base_dir, 'val'))

print("处理完成")