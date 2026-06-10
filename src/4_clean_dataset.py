import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import sys

# Windows 终端编码兼容性设置
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def is_polluted(img_path):
    # 1. 尝试读取
    # cv2.imread 处理中文路径可能有坑，用 np.fromfile 绕过
    try:
        img_data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            return True, "文件损坏或无法读取"
    except Exception as e:
        return True, "读取报错"
    
    # 2. 检查文件名是否包含明确的系统截图特征
    if "ScreenShot" in img_path.name:
        return True, "截图文件(大概率含UI或红框)"
        
    # 3. 颜色过滤：检查画面中是否有大量的“纯正红色” (旧系统的红框/红字)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色的 HSV 范围
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # 如果纯红像素过多，认定为被污染
    red_pixel_count = cv2.countNonZero(mask)
    if red_pixel_count > 1000:
        return True, f"红框文字污染"
        
    return False, "正常"

def clean_dataset():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    folders_to_check = [base_dir / "不良图片示例"]
    
    trash_dir = base_dir / "已被清理的非合规图片"
    trash_dir.mkdir(exist_ok=True)
    
    removed_count = 0
    total_count = 0
    
    for folder in folders_to_check:
        if not folder.exists(): continue
        # 遍历所有子文件夹里的图片
        for img_path in folder.rglob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg', '.bmp']:
                continue
                
            total_count += 1
            polluted, reason = is_polluted(img_path)
            if polluted:
                # 安全起见，不直接 os.remove，而是移动到隔离区
                dst = trash_dir / f"{img_path.name}"
                # 如果重名则加个后缀
                if dst.exists():
                    dst = trash_dir / f"{reason}_{img_path.name}"
                
                try:
                    shutil.move(str(img_path), str(dst))
                    removed_count += 1
                    print(f"🗑️ 已剔除: {img_path.name} -> 原因: {reason}")
                except Exception as e:
                    pass
                
    print(f"\n✅ 数据集清理完成！扫描总数: {total_count}，隔离不合规图片: {removed_count}。")
    print(f"👉 被剔除的图片已隔离至: {trash_dir}")

if __name__ == "__main__":
    clean_dataset()
