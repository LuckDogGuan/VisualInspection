# 铝型材表面缺陷检测 (Visual Inspection for Aluminum Profiles)

本项目旨在建立一套基于 YOLOv8 / YOLO11 的铝型材表面缺陷检测系统，支持级联检测流水线设计（快速粗筛 + 细分通道监控），适用于工业工控机 CPU 推理及服务器 GPU 训练。

---

## 📂 项目目录结构说明

```text
铝型材缺陷图/
├── .git/                      # Git 版本控制配置
├── .gitignore                  # Git 忽略文件（已忽略图片、权重及临时结果）
├── README.md                   # 本说明文件
├── Phase1_MVP_Requirements.md  # 第一阶段 MVP 需求与环境配置说明
├── 厂区加工流程.jpg            # 固美铝型材工艺流程看板图
├── sample_contact_sheet.jpg    # 铝型材缺陷样例图片拼图
│
├── APSPC1/                     # 原始高清原图库 1 (615张图片, imgXXXX.jpg)
├── APSPC2/                     # 原始高清原图库 2 (656张图片, imgXXXX.jpg)
├── 不良图片示例/                # 质检提取出的不合规缺陷图片示例 (已清理)
│
├── datasets/                   # 数据集根目录
│   ├── raw_annotations/        # [待放入] 存放从天池或外部下载的 XML/JSON 标注原文件
│   └── aluminum_defects/       # 自动划分好的训练/验证数据集 (由 5_voc2yolo.py 生成)
│       ├── images/             # 训练与验证图片 (train/val)
│       └── labels/             # 转换为 YOLO 格式的 txt 标签 (train/val)
│
├── 检测结果输出/                # 批量检测及通用模型推理的可视化渲染图与 CSV 报表
│   └── 通用模型检测结果/        # 通用缺陷模型检测输出
│
└── src/                        # 核心算法与工具脚本目录
    ├── yolov8n.pt              # 官方原始 YOLOv8n 权重
    ├── neu_det_best.pt         # 下载好的通用金属缺陷检测模型权重
    ├── dataset.yaml            # YOLO 训练类别与路径配置文件
    │
    ├── 0_setup_dataset_folders.py  # 数据集目录结构初始化脚本
    ├── 1_train.py              # 模型训练启动脚本（已优化为自动适配 CPU/GPU）
    ├── 2_batch_detect.py       # 文件夹批量常规检测脚本
    ├── 3_auto_label.py         # AI 辅助自动标注脚本
    ├── 4_clean_dataset.py      # 数据集清洗脚本（自动剔除红框/截图污染）
    ├── 5_voc2yolo.py           # XML标注对齐原图并自动切分训练集脚本
    ├── 6_evaluate_general_model.py # 通用缺陷模型评估与推理脚本
    └── pipeline_detect.py      # 模块化级联判定框架（初筛过滤 + 多通道分流）
```

---

## 🛠️ 核心开发工作流

### 第一步：清理截图污染（已完成）
运行 `python src/4_clean_dataset.py`。它会自动检测并把 `不良图片示例` 中被系统红框/红字污染过的截图移动到隔离区，保护模型的学习准确度。

### 第二步：评估通用模型（已完成）
运行 `python src/6_evaluate_general_model.py`。该脚本自动调用下载好的 `src/neu_det_best.pt` 通用金属表面缺陷模型在 `不良图片示例` 上进行检测。您可以进入 `检测结果输出/通用模型检测结果/` 查看带有画框的图片。
*   **结论**：该模型对“划痕 (scratches)”敏感度极高，适合拿来做 AI 辅助预标注。

### 第三步：下载标注文件并对齐划分
1. 下载天池或外部对应的标注 XML 压缩包并解压，将所有 XML 文件放入 `datasets/raw_annotations/` 中。
2. 运行 `python src/5_voc2yolo.py`。脚本会自动转换标注格式并与本地 `APSPC1/2` 文件夹里的原图配对，自动建立并划分好 `datasets/aluminum_defects/` 下的训练图片和标签。

### 第四步：模型训练
将整个工程上传至您的服务器，运行 `python src/1_train.py`。脚本将自动启用服务器 GPU 进行训练。

### 第五步：级联检测部署
将训练好的 `best.pt` 拷回本地，运行 `python src/pipeline_detect.py`，程序会开启高效的级联分流检测，并自动生成结构化的质检报告 CSV 文件。
