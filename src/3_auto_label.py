import os
from pathlib import Path
from ultralytics import YOLO

def auto_label():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    input_images_dir = base_dir / "data" / "yolo_dataset" / "images" / "train"
    weights_dir = base_dir / "src" / "runs"
    all_weights = list(weights_dir.rglob("best.pt"))
    if not all_weights:
        fallback_pt = base_dir / "src" / "neu_det_best.pt"
        if fallback_pt.exists():
            print(f"ℹ️ 未找到自定义 best.pt 权重，已加载通用金属缺陷模型 `src/neu_det_best.pt` 进行 AI 自动标注。")
            weights_path = fallback_pt
        else:
            print("⚠️ 还没有训练过模型，且未在 src/ 发现 neu_det_best.pt 通用权重！")
            return
    else:
        weights_path = max(all_weights, key=os.path.getmtime)
        print(f"✅ 自动加载最新模型: {weights_path}")

    model = YOLO(weights_path)
    print(f"🚀 开始用 AI 辅助自动标注: {input_images_dir}")
    
    results = model.predict(
        source=str(input_images_dir),
        save_txt=True, 
        save_conf=False, 
        project=str(base_dir / "src" / "runs" / "auto_label"),
        name="labels_output",
        conf=0.15 
    )
    
    output_txt_dir = base_dir / "src" / "runs" / "auto_label" / "labels_output" / "labels"
    target_lbl_dir = base_dir / "data" / "yolo_dataset" / "labels" / "train"
    target_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    txt_count = 0
    if output_txt_dir.exists():
        for txt_file in output_txt_dir.glob("*.txt"):
            target_path = target_lbl_dir / txt_file.name
            if not target_path.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                txt_count += 1
                
    print(f"✅ AI 预标注完成！共为 {txt_count} 张图片生成了初版标签。")

if __name__ == "__main__":
    auto_label()
