# v4 重复样本分析

生成时间：2026-06-24

## 结论

- 重复/近似重复组：25
- 建议删除重复样本：193
- 去重候选数据集：`data/yolo_stage3_manual/exports/dataset_clean_candidate_20260624_v4_dedup`

## 规则

1. 同一组重复/近似重复样本中，优先保留 `batch_005`。
2. 如果没有 `batch_005`，优先保留批次号更大的样本。
3. 批次相同则优先保留人工修正或合并后的样本。
4. 只在同一类别内去重，避免不同缺陷类别被误合并。
5. 图像哈希只作为辅助证据，后续人工复核以 `duplicate_groups.csv` 和 contact sheet 为准。

## 输出

- `duplicate_groups.csv`：每个重复组的保留/删除建议。
- `index.html`：重复组可视化。
- `contact_sheets/`：每组重复图拼图。
