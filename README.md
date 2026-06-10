# 铝型材表面缺陷检测 (Visual Inspection for Aluminum Profiles)

本项目是一套基于 YOLOv8 / YOLO11 的铝型材表面缺陷检测系统，支持**“级联检测流水线”**（快速粗筛 OK/NG + 多通道专用模型分流监控）设计，适用于工业工控机 CPU 现场推理及云端服务器 GPU 训练。

---

## 📂 项目最新目录结构

为了使项目更加规范，目录已被重新划分为 **`data`（输入数据）**、**`outputs`（输出结果）** 和 **`src`（算法源码）** 三个核心板块：

```text
铝型材缺陷图/
├── .gitignore                  # Git 忽略规则文件（已忽略大图、权重及临时结果）
├── README.md                   # 本说明文件
├── Phase1_MVP_Requirements.md  # 第一阶段 MVP 需求与环境配置说明
├── 厂区加工流程.jpg            # 固美铝型材工艺流程看板图
├── sample_contact_sheet.jpg    # 铝型材缺陷样例图片拼图
│
├── data/                       # 【数据输入区】所有输入图像及标注均在此处
│   ├── raw_images/             # 工厂原始监控图像库 (包含 APSPC1 和 APSPC2)
│   ├── bad_examples/           # 质检提取出的不合规缺陷图片示例目录
│   ├── raw_annotations/        # [待放入] 存放从外部下载的 XML/JSON 原始标注文件
│   └── yolo_dataset/           # 自动生成和划分好的 YOLO 训练集目录
│
├── outputs/                    # 【检测输出区】所有运行报告、画框图均在此处
│   ├── batch_results/          # 批量常规检测和级联流水线报告输出目录
│   ├── general_model_results/  # 通用模型在 bad_examples 上的画框渲染图与评估报表
│   └── trash_bin/              # 自动隔离的非合规图片（含红框/截图等污染）
│
└── src/                        # 【核心算法源码区】所有 Python 代码
    ├── yolov8n.pt              # 官方原始 YOLOv8n 权重
    ├── neu_det_best.pt         # 自动下载的通用金属表面缺陷检测模型权重
    ├── dataset.yaml            # YOLO 训练集配置与类别定义文件
    │
    ├── 0_setup_dataset_folders.py  # 训练目录结构快速初始化脚本
    ├── 1_train.py              # 模型训练脚本（已优化为自动适配 GPU/CPU）
    ├── 2_batch_detect.py       # 常规批量检测脚本
    ├── 3_auto_label.py         # AI 自动预标注脚本（自动调用通用模型）
    ├── 4_clean_dataset.py      # 数据清洗与红框过滤脚本
    ├── 5_voc2yolo.py           # XML标注对齐并自动生成训练集脚本
    ├── 6_evaluate_general_model.py # 通用缺陷模型评估与推理脚本
    ├── 7_evaluate_100_images.py # 随机抽取 100 张原图进行检测评估脚本
    ├── 8_auto_label_from_classified.py # 分类引导自动标注与划分数据集脚本
    └── pipeline_detect.py      # 模块化级联判定框架主程序
```

---

## 💻 命令行运行手册 (如何运行项目？)

所有操作均在项目根目录 `铝型材缺陷图` 下打开终端（如 PowerShell 或 Cmd）运行。

### 1. 数据清洗（过滤带有红框的系统截图）
*   **命令**：
    ```bash
    python src/4_clean_dataset.py
    ```
*   **作用**：自动检查 `data/bad_examples` 中的图片。如果图片包含系统自带红框或属于“ScreenShot”截图，会将其移动至 `outputs/trash_bin` 进行安全隔离，防止模型学错。

### 2. 测试通用模型效果（快速看效果）
*   **命令**：
    ```bash
    python src/6_evaluate_general_model.py
    ```
*   **作用**：载入已下载好的 `src/neu_det_best.pt` 金属通用模型，对 `data/bad_examples` 进行缺陷预测，并将画好红框的渲染图和 Excel/CSV 报表输出到 `outputs/general_model_results/`。您可直接打开该文件夹查看效果。

### 3. 随机抽取 100 张原图检测（广泛看效果）
*   **命令**：
    ```bash
    python src/7_evaluate_100_images.py
    ```
*   **作用**：从本地原图库 `data/raw_images/APSPC1` 中随机抽样 100 张图片，进行缺陷推理画框并输出报告至 `outputs/general_model_results_100/`，可直观对比正常反光条纹与真实缺陷在通用模型下的响应。

### 4. 导入分类数据进行“AI 自动标注”（直接搞定未标注分类数据）
*   **命令**：
    ```bash
    python src/8_auto_label_from_classified.py
    ```
*   **作用**：针对未标注但已分类好的 `data/分类数据/ali2018/` 目录。脚本会自动调用定位模型圈出瑕疵坐标，并强制将标签覆盖为对应的文件夹缺陷类型（如 0-不导电, 1-擦花, 9-脏点等），且将正常品引入作为背景负样本，自动构建出包含近 2000 张图的 YOLO 训练集。

### 5. 一键对齐本地图库并划分训练集
*   **第一步**：将您下载好的原始 XML 标注文件放入 `data/raw_annotations` 文件夹中。
*   **第二步**：在终端运行：
    ```bash
    python src/5_voc2yolo.py
    ```
*   **作用**：脚本会自动读取 XML，把坐标转化为 YOLO 归一化格式，并在本地对齐匹配 `data/raw_images` 里的 1271 张图片，按 9:1 的比例自动存入 `data/yolo_dataset` 供训练使用。

### 6. 启动模型训练（上传到服务器后运行）
*   **命令**：
    ```bash
    python src/1_train.py
    ```
*   **作用**：加载 `src/dataset.yaml` 启动 YOLO 训练。代码已做自适应优化：本地运行时自动采用 CPU 模式以验证代码；在云端服务器运行时，会自动调用显卡（GPU）进行快速训练。

### 7. 运行现场“级联检测流水线”
*   **命令**：
    ```bash
    python src/pipeline_detect.py
    ```
*   **作用**：启动工业级流水性质检程序。包含两级过滤：Stage 1 粗筛过滤（OK图直接放行，NG图留底），Stage 2 子分类器分流判定。运行后在 `outputs/batch_results` 中生成判定汇总报表。
