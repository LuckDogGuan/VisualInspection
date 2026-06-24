# Clean Candidate Dataset 20260623 v1

生成时间：2026-06-23

## 目的

这个数据集用于验证“标注质量复核和局部合并规则”是否能改善 YOLO 框选效果。它不会覆盖原始人工标注，也不会修改 `annotation_registry.csv`。

## 输入范围

```text
来源：data/yolo_stage3_manual/annotation_registry.csv
状态：status=annotated
问题批次：batch_004, batch_005
问题类别：stain, dent, powder
```

处理规则：

```text
stain / dent / powder 在 batch_004 / batch_005 中按局部相邻小框合并规则生成候选标签。
其他已标注图片原样保留。
所有导出图片使用英文文件名，完整中文原始文件名保存在 clean_manifest.csv。
```

## 导出结果

```text
总图片：265
训练集：212
验证集：53
```

分类别：

```text
dent：58
powder：60
stain：87
crack：30
transverse_bump：30
```

按处理方式：

```text
merge_candidate：72
keep：43
kept_original：150
```

## 训练入口

本地或服务器训练时使用：

```text
data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v1/data.yaml
```

训练后重点比较：

```text
整体 mAP50
stain / dent / powder mAP50
val_batch*_pred.jpg 中预测框是否比 dataset_annotated_265 更贴近真实缺陷
正常区域误检是否增加
```

## 注意

这是候选数据集，不是最终人工标注。绿色建议框来自规则化辅助合并，仍需要通过训练结果和人工观察确认是否有效。
