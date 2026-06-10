import os
import shutil
import random
from pathlib import Path

def setup_dataset():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    source_dir = base_dir / "data" / "raw_images"
    
    dataset_dir = base_dir / "data" / "yolo_dataset"
    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"✅ YOLO 训练数据集文件夹结构已创建在: {dataset_dir}")
    
    extracted_count = 0
    for subfolder in ["APSPC1", "APSPC2"]:
        src_sub = source_dir / subfolder
        if src_sub.exists():
            all_imgs = list(src_sub.glob("*.jpg"))
            if all_imgs:
                sampled_imgs = random.sample(all_imgs, min(50, len(all_imgs)))
                for img in sampled_imgs:
                    dst = train_img_dir / f"{subfolder}_{img.name}"
                    if not dst.exists():
                        shutil.copy2(img, dst)
                        extracted_count += 1
                        
    print(f"✅ 已从图库中随机抽取 {extracted_count} 张纯净原图至 {train_img_dir}")
    print("👉 下一步：请使用 AnyLabeling 或 LabelImg 打开上述 train 文件夹进行初步的人工画框标注。")

if __name__ == "__main__":
    setup_dataset()
