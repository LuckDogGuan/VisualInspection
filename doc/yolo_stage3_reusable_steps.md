# YOLO Stage 3 可复用步骤

更新时间：2026-06-24

目标：把铝型材缺陷检测的标注、复核、训练拆成小步骤。每一步都有明确输入和输出，后续只重复运行步骤，不重新写代码。

## Step 1 生成预测复核清单

用途：把人工 GT 和模型预测放在同一张图里，找出漏检、重复框、错类、框偏差大的样本。

输入：

```text
data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v3/
server_outputs/predict_clean_candidate_20260623_v3_val/
```

命令：

```powershell
python scripts\create_yolo_prediction_review.py `
  --dataset-root data\yolo_stage3_manual\exports\dataset_clean_candidate_20260623_v3 `
  --prediction-root server_outputs\predict_clean_candidate_20260623_v3_val `
  --output-dir data\yolo_stage3_manual\review\v3_prediction_review `
  --split val `
  --max-items 40
```

输出：

```text
data/yolo_stage3_manual/review/v3_prediction_review/index.html
data/yolo_stage3_manual/review/v3_prediction_review/review_priority.csv
data/yolo_stage3_manual/review/v3_prediction_review/guides/
```

复核点：

```text
优先看 score 高的样本
stain 漏检优先
dent 重复框优先
crack 大框只处理明显过大的
不确定的跳过
```

## Step 2 重复/相似图复核

用途：防止重复标注，但避免误删背景相似的工业图。

命令：

```powershell
python scripts\create_yolo_duplicate_review.py `
  --dataset-root data\yolo_stage3_manual\exports\dataset_clean_candidate_20260623_v3 `
  --output-root data\yolo_stage3_manual\exports\dataset_clean_candidate_20260624_v4_dedup_strict `
  --review-dir data\yolo_stage3_manual\review\v4_duplicate_review_strict `
  --hash-threshold 4
```

输出：

```text
data/yolo_stage3_manual/review/v4_duplicate_review_strict/index.html
data/yolo_stage3_manual/review/v4_duplicate_review_strict/similar_hash_review.csv
data/yolo_stage3_manual/review/v4_duplicate_review_strict/duplicate_review_decisions.csv
```

当前规则：

```text
自动删除只接受同类别 + 同来源文件名/路径
哈希相似只提示复核，不自动删除
同组重复优先保留 batch_005
```

当前结论：

```text
25 组疑似重复已全部 keep_all
后续人工标注不再处理 similar_xxx
```

## Step 3 生成 stain 质检工人原图判断包

用途：把 stain 单独拎出来给质检工人判断。页面必须带原图，质检以原图为准；GT/预测对照图只作为参考。

命令：

```powershell
python scripts\create_stain_manual_review_pack.py `
  --prediction-review-dir data\yolo_stage3_manual\review\v3_prediction_review `
  --dataset-root data\yolo_stage3_manual\exports\dataset_clean_candidate_20260623_v3 `
  --output-dir data\yolo_stage3_manual\review\stain_manual_review_20260624 `
  --limit 25
```

输出：

```text
data/yolo_stage3_manual/review/stain_manual_review_20260624/发给质检工人/
data/yolo_stage3_manual/review/stain_manual_review_20260624/index.html
data/yolo_stage3_manual/review/stain_manual_review_20260624/qc_worker_index.html
data/yolo_stage3_manual/review/stain_manual_review_20260624/qc_worker_reply_template.csv
data/yolo_stage3_manual/review/stain_manual_review_20260624/qc_worker_annotation_template.md
data/yolo_stage3_manual/review/stain_manual_review_20260624/stain_manual_decisions_template.csv
data/yolo_stage3_manual/review/stain_manual_review_20260624/originals/
data/yolo_stage3_manual/review/stain_manual_review_20260624/contact_sheets/
data/yolo_stage3_manual/review/stain_manual_review_20260624/guides/
```

优先发给质检工人的文件夹：

```text
data/yolo_stage3_manual/review/stain_manual_review_20260624/发给质检工人/
  01_质检说明.md
  02_质检回复模板.csv
  03_图片清单.csv
  04_查看页面.html
  原图/
  参考图/
  拼图/
```

质检工人回复格式：

```text
图片编号：stain_batch_001_0041.jpg
判断结果：需要标注 / 跳过 / 不确定
处理方式：标一个框 / 标多个框 / 合并成一个区域 / 删除不明显框 / 保留当前 / 跳过
框数：例如 2
位置说明：例如 左侧一处，中间一处，右上两处
跳过原因：缺陷太淡或看不清 / 疑似反光或拍摄影响 / 更像正常纹理或色差 / 不是缺陷 / 和其他框重复 / 不确定，需要确认
备注：
```

也可以填写：

```text
data/yolo_stage3_manual/review/stain_manual_review_20260624/发给质检工人/02_质检回复模板.csv
```

固定话术模板：

```text
doc/qc_worker_small_defect_annotation_template.md
```

## Step 4 自动保守优化实验

用途：只做明确的几何优化，先生成候选，再视觉复核。

命令：

```powershell
python scripts\auto_optimize_yolo_labels_conservative.py `
  --source-root data\yolo_stage3_manual\exports\dataset_clean_candidate_20260623_v3 `
  --output-root data\yolo_stage3_manual\exports\dataset_clean_candidate_20260624_v4_visual_conservative `
  --review-dir data\yolo_stage3_manual\review\v4_visual_conservative_review
```

当前结论：

```text
v4_visual best mAP50: 0.44949
v3 best mAP50: 0.47975
v4_visual 不作为当前基线
```

后续原则：

```text
自动优化只能做候选
必须看 before/after 图
训练指标下降则回到 v3
```

## Step 5 服务器训练

用途：训练一个导出的 YOLO 数据集。

上传：

```powershell
scp -r data/yolo_stage3_manual/exports/<dataset_name> xmu-server:/data/home/guanjianxiong/visualinspection/yolo_stage3/
scp scripts/train_yolo_stage3.py xmu-server:/data/home/guanjianxiong/visualinspection/yolo_stage3/scripts/
```

训练：

```bash
cd /data/home/guanjianxiong/visualinspection/yolo_stage3
nohup python3 scripts/train_yolo_stage3.py \
  --data <dataset_name>/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 64 \
  --device 1 \
  --workers 8 \
  --project /data/home/guanjianxiong/visualinspection/yolo_stage3/outputs \
  --name server_train_<dataset_name> \
  --no-amp > outputs/server_train_<dataset_name>.log 2>&1 < /dev/null &
```

拉回：

```powershell
scp xmu-server:/data/home/guanjianxiong/visualinspection/yolo_stage3/outputs/server_train_<dataset_name>/training_report.md `
    xmu-server:/data/home/guanjianxiong/visualinspection/yolo_stage3/outputs/server_train_<dataset_name>/results.csv `
    xmu-server:/data/home/guanjianxiong/visualinspection/yolo_stage3/outputs/server_train_<dataset_name>/results.png `
    xmu-server:/data/home/guanjianxiong/visualinspection/yolo_stage3/outputs/server_train_<dataset_name>/val_batch0_pred.jpg `
    server_outputs/server_train_<dataset_name>/
```

## Step 6 训练结果判断

当前基线：

```text
v3 best mAP50: 0.47975
v3 final mAP50: 0.45509
```

判断规则：

```text
best mAP50 明显高于 v3，才考虑替换基线
final mAP50 下降时，先看预测图是否更合理
stain 漏检没有改善，不要只因 crack/dent 视觉更好就替换基线
```

## 当前下一步

继续由人工判断 stain：

```text
data/yolo_stage3_manual/review/stain_manual_review_20260624/index.html
data/yolo_stage3_manual/review/stain_manual_review_20260624/stain_manual_decisions_template.csv
```

人工判断完成后，再生成 v5 数据集并训练。
