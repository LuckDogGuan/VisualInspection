import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def batch_detect():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    # 测试时，以 APSPC1 目录为例进行批量测试
    input_folder = base_dir / "APSPC1" 
    output_folder = base_dir / "检测结果输出"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    weights_dir = base_dir / "src" / "runs"
    all_weights = list(weights_dir.rglob("best.pt"))
    if not all_weights:
        print("⚠️ 没找到训练好的 best.pt，加载默认 yolov8n.pt 仅作代码测试流程。")
        weights_path = 'yolov8n.pt'
    else:
        weights_path = max(all_weights, key=os.path.getmtime)
        print(f"✅ 自动加载最新模型: {weights_path}")
        
    model = YOLO(weights_path)
    print(f"🚀 开始批量检测文件夹: {input_folder}")
    
    results = model.predict(
        source=str(input_folder),
        save=True, 
        project=str(output_folder),
        name="batch_1",
        conf=0.25
    )
    
    data_list = []
    for r in results:
        img_path = Path(r.path)
        img_name = img_path.name
        boxes = r.boxes
        if len(boxes) == 0:
            data_list.append({
                "图片名称": img_name,
                "判定结果": "OK",
                "缺陷数量": 0,
                "详细坐标_类别_置信度": ""
            })
        else:
            details = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                conf = float(box.conf[0])
                xyxy = [round(v, 1) for v in box.xyxy[0].tolist()] 
                details.append(f"[{cls_name} ({(conf*100):.1f}%) 坐标:{xyxy}]")
                
            data_list.append({
                "图片名称": img_name,
                "判定结果": "NG",
                "缺陷数量": len(boxes),
                "详细坐标_类别_置信度": " | ".join(details)
            })
            
    df = pd.DataFrame(data_list)
    report_path = output_folder / "批量检测报告.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 批量检测完成！")
    print(f"👉 画框后的图片保存在: {output_folder / 'batch_1'}")
    print(f"👉 判定数据报表保存在: {report_path}")

if __name__ == "__main__":
    batch_detect()
