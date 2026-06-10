import os
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt') 
    yaml_path = r"d:\code\VisualInspection\铝型材缺陷图\src\dataset.yaml"
    
    print("🚀 开始训练模型...")
    results = model.train(
        data=yaml_path,
        epochs=30,             
        imgsz=640,             
        batch=8,               
        device=None,           # 自动检测 GPU（如有则用显卡，无则用CPU）
        project='runs/train',  
        name='defect_v1'       
    )
    
    print("✅ 训练完成！权重文件保存在 src/runs/train/defect_v1/weights/best.pt")

if __name__ == '__main__':
    main()
