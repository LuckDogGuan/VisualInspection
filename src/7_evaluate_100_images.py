import os
import random
from pathlib import Path
from ultralytics import YOLO
import cv2
import pandas as pd
import sys

# Windows 终端编码兼容性设置
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    model_path = base_dir / "src" / "neu_det_best.pt"
    
    if not model_path.exists():
        print("❌ 未在 src/ 目录下找到 neu_det_best.pt。请先确保下载了通用权重。")
        return
        
    # 1. 寻找 APSPC1 目录
    img_dir = base_dir / "data" / "raw_images" / "APSPC1"
    if not img_dir.exists():
        print(f"❌ 未找到大图目录: {img_dir}")
        return
        
    all_imgs = list(img_dir.glob("*.jpg"))
    if not all_imgs:
        print(f"❌ {img_dir} 目录下没有找到任何 .jpg 图片。")
        return
        
    print(f"📂 发现总计 {len(all_imgs)} 张原始图片。")
    
    # 2. 随机抽取 100 张
    sample_size = min(100, len(all_imgs))
    random.seed(42)  # 固定随机种子确保可复现
    sampled_imgs = random.sample(all_imgs, sample_size)
    print(f"🎲 已成功随机抽取 {sample_size} 张图片进行推理检测。")
    
    # 3. 载入模型并创建输出目录
    model = YOLO(str(model_path))
    output_dir = base_dir / "outputs" / "general_model_results_100"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 开始批量推理检测图片，渲染图将保存至: {output_dir} ...")
    
    records = []
    processed_count = 0
    ng_count = 0
    
    for img_path in sampled_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # 运行检测 (置信度设为 0.25)
        results = model.predict(source=img, conf=0.25, verbose=False)
        r = results[0]
        boxes = r.boxes
        
        detected_details = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            
            detected_details.append(f"[{cls_name} ({(conf*100):.1f}%) 坐标:{xyxy}]")
            
            # 绘制边界框与标签
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            cv2.putText(img, f"{cls_name} {(conf*100):.0f}%", (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        has_defect = len(boxes) > 0
        if has_defect:
            ng_count += 1
            # 只有检测到缺陷时，才保存渲染画框后的图片，节省空间
            save_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(save_path), img)
            
        records.append({
            "原图名称": img_path.name,
            "判定结果": "NG" if has_defect else "OK",
            "检出缺陷数": len(boxes),
            "缺陷详情": " | ".join(detected_details) if has_defect else "无缺陷"
        })
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"⏳ 已处理: {processed_count} / {sample_size} 张图片...")
            
    # 保存 CSV 报告
    df = pd.DataFrame(records)
    report_path = output_dir / "evaluation_report_100.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print("\n🎉 随机 100 张图片检测评估完成！")
    print(f"📊 统计结果：")
    print(f"   ├─ 扫描总数: {processed_count} 张")
    print(f"   ├─ 判定 NG (有瑕疵): {ng_count} 张 (比例: {(ng_count/processed_count*100):.1f}%)")
    print(f"   └─ 判定 OK (合格品): {processed_count - ng_count} 张")
    print(f"👉 判定数据报表已保存至: {report_path}")
    print(f"👉 检测到瑕疵的渲染画框图已存入: {output_dir}")

if __name__ == "__main__":
    main()
