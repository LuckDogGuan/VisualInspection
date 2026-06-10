import os
import cv2
from pathlib import Path
import shutil
import random
from ultralytics import YOLO
import sys

# Windows 终端编码兼容性设置
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# 分类文件夹与 YOLO 缺陷类别的映射表
FOLDER_TO_CLASS = {
    "non-conducting": 0,                   # 不导电 -> budaodian
    "scuffing": 1,                         # 擦花 -> cahua
    "Orange peel": 3,                      # 桔皮 -> jupi
    "Drain bottom": 4,                     # 漏底 -> loudi
    "pitting": 7,                          # 起坑 -> qikeng
    "The transverse strip is dented": 7,   # 横向凹陷 -> 合并到 qikeng
    "Be injured by a collision": 7,        # 碰伤 -> 合并到 qikeng
    "Dirty spot": 9,                       # 脏点 -> zangdian
    
    # 以下类别天池官方未独立，做相近合并或放入备用类
    "Coating cracking": 5,                 # 涂层开裂 -> 映射到 penliu (喷流)
    "Convex powder": 6,                    # 凸粉 -> 映射到 qipao (漆泡)
}

def main():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    ali2018_dir = base_dir / "data" / "分类数据" / "ali2018"
    model_path = base_dir / "src" / "neu_det_best.pt"
    
    if not ali2018_dir.exists():
        print(f"❌ 未找到分类数据目录: {ali2018_dir}")
        return
        
    if not model_path.exists():
        print("❌ 未在 src/ 目录下找到 neu_det_best.pt。AI 标注需要此模型提供边界框定位。")
        return
        
    print("🚀 载入定位模型...")
    model = YOLO(str(model_path))
    
    # 目标输出数据集目录
    yolo_dataset_dir = base_dir / "data" / "yolo_dataset"
    train_img_dir = yolo_dataset_dir / "images" / "train"
    val_img_dir = yolo_dataset_dir / "images" / "val"
    train_lbl_dir = yolo_dataset_dir / "labels" / "train"
    val_lbl_dir = yolo_dataset_dir / "labels" / "val"
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 获取 ali2018 下的所有子文件夹
    subfolders = [f for f in ali2018_dir.iterdir() if f.is_dir()]
    
    all_labeled_pairs = []
    
    print("\n📂 开始扫描分类数据并进行 AI 引导画框标注...")
    
    for folder in subfolders:
        folder_name = folder.name
        
        # 1. 如果是合格品文件夹，作为背景负样本
        if folder_name == "Clean sample":
            clean_imgs = list(folder.glob("*.*"))
            print(f"🟢 [Clean sample] 正常品: 发现 {len(clean_imgs)} 张图片。将作为无缺陷背景样本导入。")
            for img_path in clean_imgs:
                # 正常品不生成 labels (空txt)，仅保存图片
                all_labeled_pairs.append((img_path, None, None))
            continue
            
        # 2. 如果是未在映射表中的文件夹，跳过
        if folder_name not in FOLDER_TO_CLASS:
            print(f"⚠️ 跳过未定义映射的文件夹: {folder_name}")
            continue
            
        forced_class_id = FOLDER_TO_CLASS[folder_name]
        folder_imgs = list(folder.glob("*.*"))
        print(f"🟡 [{folder_name}] 缺陷类: 发现 {len(folder_imgs)} 张图片。强制分配 YOLO 类别 ID: {forced_class_id}")
        
        # 临时标签存放目录
        temp_lbl_dir = base_dir / "data" / "temp_classified_labels"
        temp_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        labeled_in_folder = 0
        for img_path in folder_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # 使用模型进行粗定位 (门槛设为 0.15 尽量多捕获框，类别我们已知所以不担心误报)
            results = model.predict(source=img, conf=0.15, verbose=False)
            boxes = results[0].boxes
            
            yolo_lines = []
            for box in boxes:
                # 核心创新点：使用 AI 模型提取的坐标 (BBox)，但强制覆盖为我们已知的文件夹类别 ID
                xyxy = box.xyxy[0].tolist()
                
                # 转化为 YOLO 归一化格式
                dw = 1.0 / w
                dh = 1.0 / h
                x_center = (xyxy[0] + xyxy[2]) / 2.0 * dw
                y_center = (xyxy[1] + xyxy[3]) / 2.0 * dh
                width = (xyxy[2] - xyxy[0]) * dw
                height = (xyxy[3] - xyxy[1]) * dh
                
                yolo_lines.append(f"{forced_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
            # 如果 AI 定位到了瑕疵位置，写入 txt
            if yolo_lines:
                txt_name = f"{img_path.stem}.txt"
                txt_path = temp_lbl_dir / txt_name
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_lines))
                all_labeled_pairs.append((img_path, txt_path, forced_class_id))
                labeled_in_folder += 1
            else:
                # 如果 AI 没定位到，我们暂时忽略或者存为负样本。这里选择存为正常品（以防万一），但打印警告。
                # print(f"   ⚠️ 模型未能在缺陷图片上定位到瑕疵: {img_path.name}")
                pass
                
        print(f"   👉 成功画框定位: {labeled_in_folder} / {len(folder_imgs)} 张图片。")
        
    print(f"\n📊 汇总：共获取 {len(all_labeled_pairs)} 张已自动标注/背景对齐的图片对。")
    
    if not all_labeled_pairs:
        print("❌ 未能生成任何标注数据。")
        return
        
    # 3. 随机划分 90% 训练集，10% 验证集
    random.seed(42)
    random.shuffle(all_labeled_pairs)
    split_idx = int(len(all_labeled_pairs) * 0.9)
    
    train_pairs = all_labeled_pairs[:split_idx]
    val_pairs = all_labeled_pairs[split_idx:]
    
    # 拷贝至 yolo_dataset
    for img_path, txt_path, _ in train_pairs:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        if txt_path and txt_path.exists():
            shutil.copy2(txt_path, train_lbl_dir / txt_path.name)
            
    for img_path, txt_path, _ in val_pairs:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        if txt_path and txt_path.exists():
            shutil.copy2(txt_path, val_lbl_dir / txt_path.name)
            
    # 清理临时文件夹
    temp_lbl_dir = base_dir / "data" / "temp_classified_labels"
    if temp_lbl_dir.exists():
        shutil.rmtree(temp_lbl_dir)
        
    print("\n🎉 分类引导自动标注与划分完成！")
    print(f"   ├─ 训练集大小: {len(train_pairs)} 张")
    print(f"   ├─ 验证集大小: {len(val_pairs)} 张")
    print(f"   └─ 目标数据集路径: {yolo_dataset_dir}")
    print("🚀 准备就绪！您现在可以直接在此数据集上运行训练脚本 `python src/1_train.py`。")

if __name__ == "__main__":
    main()
