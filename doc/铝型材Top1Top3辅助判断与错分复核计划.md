# 铝型材 Top1/Top3 辅助判断与错分复核计划

更新时间：2026-06-18

## 1. 目标

当前不继续盲目训练，先把模型已经出现的错分、低置信度、Top1/Top2 接近样本整理出来，交给人工复核。

核心目标：

- 提高 Top1 分类效果。
- 降低缺陷判正常的漏检风险。
- 优先优化 4 个易错类别：擦花、碰伤、凸粉、脏点。
- 用 Top3 辅助人工判断，减少单一 Top1 错判风险。

## 2. 当前推荐模型

当前优先使用服务器实验中的 Focal Loss 模型生成复核包：

```text
outputs/classification_results/exp_focal/deploy/classifier.torchscript.pt
```

原因：

- Top1：92.59%，略高于 Baseline。
- Top3：99.66%，略高于 Baseline。
- 擦花从 57.05% 提升到 75.17%。
- 正常误报与 Baseline 接近。
- 主要问题是脏点下降和 raw_images 正常候选变多，需要人工复核。

## 3. Top1/Top3 辅助判断规则

默认阈值：

```text
高置信度阈值：0.85
低置信度阈值：0.60
Top1/Top2 接近阈值：0.15
```

判断规则：

| 情况 | 处理方式 |
| --- | --- |
| Top1 置信度 >= 0.85 且 Top1-Top2 >= 0.15 | 直接采用 Top1 |
| Top1 置信度 < 0.60 | 标记低置信度，人工复核 |
| Top1-Top2 < 0.15 | 标记易混淆，人工复核 |
| Top1 置信度在 0.60 到 0.85 之间 | 输出 Top3，人工辅助判断 |

## 4. 服务器运行命令

在服务器项目目录执行：

```bash
cd /home/guanjianxiong/code/yolo/VisualInspection

/opt/miniconda3/bin/python src/classification/build_topk_review_package.py \
  --labeled-csv outputs/classification_results/exp_focal/full_screening/all_labeled_predictions.csv \
  --raw-csv outputs/classification_results/exp_focal/raw_images_quality/raw_predictions.csv \
  --output-dir outputs/classification_results/topk_review_focal \
  --high-confidence 0.85 \
  --low-confidence 0.60 \
  --close-margin 0.15
```

输出：

```text
outputs/classification_results/topk_review_focal/summary.md
outputs/classification_results/topk_review_focal/review_index.csv
outputs/classification_results/topk_review_focal/labeled_errors/
outputs/classification_results/topk_review_focal/topk_assisted/
outputs/classification_results/topk_review_focal/raw_images_review/
```

## 5. 人工复核优先级

优先级从高到低：

1. `raw_images_review/01_predicted_normal_must_review`
2. `labeled_errors/01_defect_as_normal`
3. `labeled_errors/03_focus_four_classes`
4. `topk_assisted/close_top1_top2`
5. `topk_assisted/low_confidence`
6. `raw_images_review/02_close_top1_top2`
7. `raw_images_review/02_low_confidence`

## 6. 人工复核后的用途

人工复核完成后，按结果整理：

```text
confirmed_correct/
label_wrong/
model_wrong/
uncertain/
mixed_defect/
bad_image/
```

后续训练只使用人工确认后的样本：

- 标签错的样本：修正标签后再训练。
- 模型错的样本：加入 hard sample 列表，提高采样概率。
- 不确定样本：暂不用于训练。
- 混合缺陷：单独保留，后续做多标签或裁剪策略。
- 坏图：排除。

## 7. 后续实验方向

人工复核后再进入下一轮实验：

1. Hard Sample Weighted Sampler。
2. Focal Loss + Hard Sample。
3. EfficientNet-B3。
4. ConvNeXt-Tiny。
5. Top1/Top3 线上辅助判断规则验证。

## 8. 当前注意事项

- 所有正式推理和评估都在服务器执行。
- 本地只下载图片进行人工复核，不用本地结果判断模型好坏。
- 文件名中已经包含真实类别、预测类别、置信度、Top3，人工复核时不要改原文件名。
- 原始图片不移动、不删除。
