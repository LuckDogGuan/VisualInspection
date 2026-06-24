# stain 人工判断包

生成时间：2026-06-24

## 文件

- `index.html`：按优先级查看 stain GT / Prediction 对照图。
- `qc_worker_index.html`：给质检工人看的原图判断页面。
- `qc_worker_reply_template.csv`：质检工人固定回复表。
- `stain_manual_decisions_template.csv`：人工判断模板。
- `contact_sheets/`：分组拼图，适合快速浏览。
- `originals/`：原图，质检判断以这里为准。
- `guides/`：单张对照图。

## 样本范围

- stain 待判断样本：17
- 来源：`data/yolo_stage3_manual/review/v3_prediction_review/review_priority.csv`

## 判断原则

1. 只改明显问题，不确定就写 `skip`。
2. 连续但清晰的 stain 区域可以合并成稳定区域框。
3. 分散且明显的小点可以保留多个框。
4. 框里看不出缺陷的小点删除或跳过。
5. 不要为了减少框数而把大片正常区域包进去。

## 回复格式

```text
stain_batch_001_0041：合并成左/中/右 3 个区域；很淡的小点删除。
stain_batch_001_0028：整张不明显，跳过。
stain_batch_004_0035：GT1-GT3 合并，GT4 保留，GT5 删除。
```

我拿到你的判断后，会把它转成可训练的 v5 数据集并上服务器训练。
