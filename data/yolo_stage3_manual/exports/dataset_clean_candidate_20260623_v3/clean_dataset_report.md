# Clean Candidate Dataset 20260623 v3

生成时间：2026-06-23

## 目的

这个版本在 v1 自动局部合并规则基础上，加入人工复核反馈，作为下一次 YOLO 训练的当前推荐候选数据集。

## 人工反馈来源

```text
data/yolo_stage3_manual/review/manual_review_decisions.csv
```

已应用的人工反馈包括：

```text
review_001：new-6 太大，拆回局部小框
review_003：new1 和 new2 合并
review_006：只保留 old3
review_008：删除 new5
review_011：删除 new5、new2，new4 和 new3 合并
review_013：删除 new2、new3、new5
review_014：整图 skip
review_015：删除 new1、new4
review_016：删除 new5、new2、new3，new4 放大 1.35 倍
```

## 导出结果

```text
总图片：264
训练集：211
验证集：53
标签行数：806
标签格式错误：0
```

分类别：

```text
dent：58
powder：59
stain：87
crack：30
transverse_bump：30
```

按处理方式：

```text
merge_candidate：63
manual decision：9
keep：43
kept_original：150
```

## 训练入口

```text
data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v3/data.yaml
```

## 对比对象

优先与下面两个数据集比较：

```text
data/yolo_stage3_manual/exports/dataset_annotated_265
data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v1
```

## 验收重点

```text
整体 mAP50 是否回升
stain / dent / powder 是否改善
val_batch*_pred.jpg 中框是否更接近人工想要的区域
是否因为合并框导致正常区域误检增加
```
