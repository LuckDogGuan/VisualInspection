import os
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
        print("❌ 未在 src/ 目录下找到 neu_det_best.pt，请重新运行或检查下载。")
        return
        
    print("🚀 载入通用金属缺陷检测模型 (NEU-DET v8n)...")
    model = YOLO(str(model_path))
    
    # 打印模型支持的类别
    print("\n🔍 该通用模型支持检测以下 6 种缺陷类别：")
    for idx, name in model.names.items():
        print(f"  类别 {idx}: {name}")
        
    # 定义测试输入与输出
    test_img_dir = base_dir / "data" / "bad_examples"
    output_dir = base_dir / "outputs" / "general_model_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_img_dir.exists():
        print(f"❌ 未找到测试文件夹: {test_img_dir}")
        return
        
    test_imgs = list(test_img_dir.glob("*.*"))[:8]
    if not test_imgs:
        print("❌ 测试文件夹内没有找到任何图片。")
        return
        
    print(f"\n🧪 提取 {len(test_imgs)} 张图片进行通用模型推理评估...")
    
    evaluation_records = []
    
    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # 运行推理 (低阈值 0.2，以便多抓取疑似瑕疵进行分析)
        results = model.predict(source=img, conf=0.20, verbose=False)
        r = results[0]
        boxes = r.boxes
        
        detected_details = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            
            detected_details.append(f"[{cls_name} ({(conf*100):.1f}%) 坐标:{xyxy}]")
            
            # 在图像上绘制框
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            cv2.putText(img, f"{cls_name} {(conf*100):.0f}%", (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        # 保存画框图
        save_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(save_path), img)
        
        has_defect = len(boxes) > 0
        evaluation_records.append({
            "图片名称": img_path.name,
            "检测结果": "NG" if has_defect else "OK",
            "检出缺陷数": len(boxes),
            "缺陷明细": " | ".join(detected_details) if has_defect else "未检测到缺陷"
        })
        
        print(f"📸 图像: {img_path.name} -> 检测到: {len(boxes)} 处瑕疵 | 结果: {'NG' if has_defect else 'OK'}")
        
    df = pd.DataFrame(evaluation_records)
    report_path = output_dir / "通用模型评估报告.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 评估完成！")
    print(f"👉 标注渲染图已保存至: {output_dir}")
    print(f"👉 结构化数据报表已保存至: {report_path}")
    print("\n👉 建议：打开 '通用模型检测结果' 文件夹查看渲染后的图片，分析模型是否对您的铝型材划痕有正确响应。")

if __name__ == "__main__":
    main()
