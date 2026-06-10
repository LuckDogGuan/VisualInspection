import os
import time
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
import sys

# Windows 终端编码兼容性设置
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

class BaseDetector:
    """所有独立缺陷检测器和过滤阶段的基类 (标准可复用接口)"""
    def __init__(self, model_path, name="BaseDetector", conf=0.25):
        self.name = name
        self.conf = conf
        # 兼容处理未训练好的模型路径，自动回退到官方预训练模型
        if not Path(model_path).exists():
            fallback_pt = Path(__file__).parent / 'yolov8n.pt'
            if fallback_pt.exists():
                print(f"⚠️ [{self.name}] 模型文件未找到: {model_path}，已加载本地 `src/yolov8n.pt` 进行测试。")
                self.model = YOLO(str(fallback_pt))
            else:
                print(f"⚠️ [{self.name}] 模型文件未找到: {model_path}，使用默认 yolov8n.pt 初始化以做测试。")
                self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(str(model_path))
            
    def detect(self, img):
        """输入 OpenCV 图像，返回结构化缺陷信息"""
        raise NotImplementedError("子类必须实现 detect 方法")


class Stage1FilterDetector(BaseDetector):
    """阶段 1：快速 OK/NG 初筛器
    核心目标是高召回率（零漏检），快速区分合格品与非合格品。
    """
    def detect(self, img):
        results = self.model.predict(source=img, conf=self.conf, verbose=False)
        r = results[0]
        boxes = r.boxes
        
        is_ok = len(boxes) == 0
        details = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            details.append({
                "class_id": cls_id,
                "class_name": self.model.names[cls_id] if cls_id in self.model.names else f"class_{cls_id}",
                "conf": conf,
                "xyxy": xyxy
            })
            
        return {
            "is_ok": is_ok,
            "defects": details
        }


class SpecializedChannelDetector(BaseDetector):
    """阶段 2：专用缺陷检测通道（用于精细化分类与标准模型复用）
    支持过滤特定的缺陷类别，只关注感兴趣的类别。
    """
    def __init__(self, model_path, name="SpecializedDetector", target_classes=None, conf=0.25):
        super().__init__(model_path, name, conf)
        # target_classes 是一个整数列表，表示此检测通道关心的类别索引
        self.target_classes = target_classes if target_classes is not None else []
        
    def detect(self, img):
        results = self.model.predict(source=img, conf=self.conf, verbose=False)
        r = results[0]
        boxes = r.boxes
        
        filtered_defects = []
        for box in boxes:
            cls_id = int(box.cls[0])
            # 如果配置了目标关注类别，则过滤非目标类别
            if self.target_classes and cls_id not in self.target_classes:
                continue
                
            conf = float(box.conf[0])
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            filtered_defects.append({
                "class_id": cls_id,
                "class_name": self.model.names[cls_id] if cls_id in self.model.names else f"class_{cls_id}",
                "conf": conf,
                "xyxy": xyxy
            })
            
        return {
            "defects": filtered_defects
        }


class AluminumInspectionPipeline:
    """级联检测流水线核心类"""
    def __init__(self):
        self.filter_detector = None
        self.channel_detectors = []
        
    def set_filter_stage(self, detector: Stage1FilterDetector):
        """设置 Stage 1 筛查器"""
        self.filter_detector = detector
        
    def add_channel_detector(self, detector: SpecializedChannelDetector):
        """添加 Stage 2 专用检测器"""
        self.channel_detectors.append(detector)
        
    def inspect_image(self, img_path):
        """执行级联检测"""
        start_time = time.time()
        
        # 1. 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            return {
                "图片名称": Path(img_path).name,
                "判定结果": "ERROR",
                "缺陷数量": 0,
                "判定详情": "无法读取图片文件",
                "耗时_毫秒": 0
            }
            
        # 2. Stage 1：粗筛
        if self.filter_detector is None:
            # 没有设置筛查器，直接流向下一阶段
            filter_res = {"is_ok": False, "defects": []}
        else:
            filter_res = self.filter_detector.detect(img)
            
        # 如果第一阶段判定为 OK，直接放行，结束判定
        if filter_res["is_ok"]:
            end_time = time.time()
            return {
                "图片名称": Path(img_path).name,
                "判定结果": "OK",
                "缺陷数量": 0,
                "判定详情": "Stage1 判定为合格 (OK)",
                "耗时_毫秒": int((end_time - start_time) * 1000)
            }
            
        # 3. Stage 2：进入专用检测通道分流判定
        all_detected_defects = []
        
        # 如果通道检测器列表为空，则直接使用 Stage 1 的检测结果
        if not self.channel_detectors:
            all_detected_defects = filter_res["defects"]
        else:
            for detector in self.channel_detectors:
                channel_res = detector.detect(img)
                all_detected_defects.extend(channel_res["defects"])
                
        # 4. 汇总判定结果
        is_ng = len(all_detected_defects) > 0
        end_time = time.time()
        
        details = []
        for d in all_detected_defects:
            details.append(f"[{d['class_name']} ({(d['conf']*100):.1f}%) 坐标:{d['xyxy']}]")
            
        return {
            "图片名称": Path(img_path).name,
            "判定结果": "NG" if is_ng else "OK",
            "缺陷数量": len(all_detected_defects),
            "判定详情": " | ".join(details) if is_ng else "经过各子检测器判定无目标类型缺陷 (OK)",
            "耗时_毫秒": int((end_time - start_time) * 1000)
        }


def main():
    base_dir = Path(r"d:\code\VisualInspection\铝型材缺陷图")
    src_dir = base_dir / "src"
    
    # 模拟使用的模型权重路径 (实际运行时请替换为服务器训练好的权重 best.pt)
    best_model_path = src_dir / "runs" / "train" / "defect_v1" / "weights" / "best.pt"
    
    print("🚀 正在构建级联检测流水线（Standard Cascaded Pipeline）...")
    pipeline = AluminumInspectionPipeline()
    
    # 初始化 Stage 1：快速筛查器 (低门槛阈值 0.15，防止任何缺陷漏网)
    stage1 = Stage1FilterDetector(best_model_path, name="Stage1_Filter", conf=0.15)
    pipeline.set_filter_stage(stage1)
    
    # 初始化 Stage 2 的各个专用检测通道 (这里使用对应的类别索引进行模拟关注)
    # 天池类别：1: cahua (擦花), 7: qikeng (起坑)
    damage_channel = SpecializedChannelDetector(
        best_model_path, 
        name="Stage2_Damage_Channel", 
        target_classes=[1, 7], 
        conf=0.25
    )
    
    # 天池类别：4: loudi (漏底), 5: penliu (喷流), 6: qipao (漆泡)
    coating_channel = SpecializedChannelDetector(
        best_model_path, 
        name="Stage2_Coating_Channel", 
        target_classes=[4, 5, 6], 
        conf=0.25
    )
    
    pipeline.add_channel_detector(damage_channel)
    pipeline.add_channel_detector(coating_channel)
    
    print("✅ 级联检测流水线挂载成功！")
    
    # 执行模拟测试，从不良图片示例中挑选 5 张图片进行测试运行
    test_folder = base_dir / "data" / "bad_examples"
    if not test_folder.exists() or len(list(test_folder.glob("*.*"))) == 0:
        print("⚠️ 未找到 [data/bad_examples] 文件夹或文件夹内没有图片，无法运行基准测试。")
        return
        
    test_imgs = list(test_folder.glob("*.*"))[:5]
    print(f"\n🧪 提取 {len(test_imgs)} 张图片进行流水线代码跑通验证...")
    
    report_data = []
    for img_path in test_imgs:
        res = pipeline.inspect_image(img_path)
        report_data.append(res)
        print(f"📸 图像: {res['图片名称']} -> 结果: {res['判定结果']} (缺陷数: {res['缺陷数量']}) | 耗时: {res['耗时_毫秒']}ms")
        
    df = pd.DataFrame(report_data)
    out_dir = base_dir / "outputs" / "batch_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "级联检测报告.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 验证成功！级联检测报告已保存至: {report_path}")

if __name__ == "__main__":
    main()
