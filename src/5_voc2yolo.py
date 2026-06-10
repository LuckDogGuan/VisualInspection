import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import sys

# Windows 终端编码兼容性设置
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# 类别映射表：支持天池原始拼音标签、中文别名等
LABEL_MAPPING = {
    # 拼音标签映射
    "budaodian": 0,      # 不导电
    "cahua": 1,          # 擦花
    "jiaoweiloudi": 2,   # 角位漏底
    "jupi": 3,           # 桔皮
    "loudi": 4,          # 漏底
    "penliu": 5,         # 喷流
    "qipao": 6,          # 漆泡
    "qikeng": 7,         # 起坑
    "zase": 8,           # 杂色
    "zangdian": 9,       # 脏点
    
    # 中文标签映射
    "不导电": 0,
    "擦花": 1,
    "角位漏底": 2,
    "桔皮": 3,
    "漏底": 4,
    "喷流": 5,
    "漆泡": 6,
    "起坑": 7,
    "杂色": 8,
    "脏点": 9,
    
    # 常见物理损伤合并别名
    "划痕": 1,
    "擦伤": 1,
    "碰伤": 7,
    "凹陷": 7,
    "脏污": 9
}

def convert_box_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def process_xml(xml_path, output_txt_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            return False
            
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w == 0 or h == 0:
            return False
            
        yolo_lines = []
        for obj in root.iter('object'):
            name = obj.find('name').text.strip().lower()
            if name not in LABEL_MAPPING:
                continue
            
            class_id = LABEL_MAPPING[name]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            
            bb = convert_box_to_yolo((w, h), b)
            yolo_lines.append(f"{class_id} " + " ".join([f"{val:.6f}" for val in bb]))
            
        if yolo_lines:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))
            return True
    except Exception as e:
        print(f"❌ 解析 XML 报错 {xml_path.name}: {e}")
    return False

def main():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    raw_anno_dir = base_dir / "data" / "raw_annotations"
    
    # 创建标注输入区
    raw_anno_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(raw_anno_dir.glob("*.xml"))
    if not xml_files:
        print(f"⚠️ 没有在目录 {raw_anno_dir} 找到 XML 标注文件。")
        print("👉 请先下载标注 zip 并解压所有 XML 文件到上述文件夹内，然后重新运行此脚本。")
        return
        
    print(f"📂 找到 {len(xml_files)} 个 XML 标注文件。开始对齐本地原图并转换为 YOLO 格式...")
    
    # 输出的目标文件夹
    output_dataset_dir = base_dir / "data" / "yolo_dataset"
    train_img_dir = output_dataset_dir / "images" / "train"
    val_img_dir = output_dataset_dir / "images" / "val"
    train_lbl_dir = output_dataset_dir / "labels" / "train"
    val_lbl_dir = output_dataset_dir / "labels" / "val"
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 原图搜索来源
    image_sources = [
        base_dir / "data" / "raw_images" / "APSPC1", 
        base_dir / "data" / "raw_images" / "APSPC2"
    ]
    
    matched_pairs = []
    skipped_count = 0
    
    # 用临时目录存放生成的 yolo 标注 txt 文件
    temp_txt_dir = base_dir / "data" / "temp_txt_labels"
    temp_txt_dir.mkdir(parents=True, exist_ok=True)
    
    for xml_file in xml_files:
        # XML 文件对应的图片名称 (通常是同名，但后缀是 .jpg)
        base_name = xml_file.stem
        img_name = f"{base_name}.jpg"
        
        # 寻找对应的本地原图
        found_img_path = None
        for src_dir in image_sources:
            test_path = src_dir / img_name
            if test_path.exists():
                found_img_path = test_path
                break
                
        if not found_img_path:
            skipped_count += 1
            continue
            
        # 转换并写入临时 txt
        temp_txt_path = temp_txt_dir / f"{base_name}.txt"
        if process_xml(xml_file, temp_txt_path):
            matched_pairs.append((found_img_path, temp_txt_path))
            
    print(f"🔍 原图匹配成功: {len(matched_pairs)} 组，未匹配到本地图片的标注: {skipped_count} 个。")
    
    if not matched_pairs:
        print("❌ 转换失败，未找到任何可对齐的数据对。")
        # 清理临时文件夹
        shutil.rmtree(temp_txt_dir)
        return
        
    # 打乱并按照 9:1 比例划分训练集和验证集
    random.seed(42)
    random.shuffle(matched_pairs)
    split_index = int(len(matched_pairs) * 0.9)
    
    train_pairs = matched_pairs[:split_index]
    val_pairs = matched_pairs[split_index:]
    
    # 拷贝文件到目标位置
    for img_path, txt_path in train_pairs:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        shutil.copy2(txt_path, train_lbl_dir / txt_path.name)
        
    for img_path, txt_path in val_pairs:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        shutil.copy2(txt_path, val_lbl_dir / txt_path.name)
        
    # 清理临时文件夹
    shutil.rmtree(temp_txt_dir)
    
    print("\n🎉 数据集格式转换与对齐成功！")
    print(f"👉 训练集大小: {len(train_pairs)} 组")
    print(f"👉 验证集大小: {len(val_pairs)} 组")
    print(f"👉 目标训练根目录: {output_dataset_dir}")
    print("🚀 您现在可以把整个工程拷到服务器，运行 `python src/1_train.py` 开始训练模型了！")

if __name__ == "__main__":
    main()
