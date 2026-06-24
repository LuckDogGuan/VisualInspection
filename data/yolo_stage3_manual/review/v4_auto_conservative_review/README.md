# v4 自动保守标注优化

生成时间：2026-06-24

## 结论

- 输出数据集：`data/yolo_stage3_manual/exports/dataset_clean_candidate_20260624_v4_auto_conservative`
- 总样本：264
- 自动修改样本：39

## 修改类型

- crack_shrink_tall_wide_boxes: 19
- dent_merge_duplicate_boxes: 7
- stain_merge_dense_regions: 13

## 保守规则

1. `stain`：只合并连续/密集的小框；合并后区域过大则跳过。
2. `dent`：只合并重叠或极近的小框；分散缺陷保留分开。
3. `crack`：只收紧很宽且明显过高的大框；不确定框原样保留。
4. `powder`、`transverse_bump`：本轮不自动改，避免扩展不稳定类别。

## 已修改样本示例

- `dent_batch_005_0002.jpg`: dent_merge_duplicate_boxes, 6 -> 5
- `dent_batch_003_0010.jpg`: dent_merge_duplicate_boxes, 10 -> 6
- `dent_batch_001_0017.jpg`: dent_merge_duplicate_boxes, 3 -> 2
- `dent_batch_005_0022.jpg`: dent_merge_duplicate_boxes, 5 -> 4
- `dent_batch_005_0035.jpg`: dent_merge_duplicate_boxes, 3 -> 2
- `dent_batch_005_0037.jpg`: dent_merge_duplicate_boxes, 3 -> 2
- `dent_batch_005_0045.jpg`: dent_merge_duplicate_boxes, 3 -> 2
- `stain_batch_004_0096.jpg`: stain_merge_dense_regions, 5 -> 3
- `stain_batch_001_0098.jpg`: stain_merge_dense_regions, 8 -> 6
- `stain_batch_004_0099.jpg`: stain_merge_dense_regions, 4 -> 2
- `stain_batch_004_0112.jpg`: stain_merge_dense_regions, 5 -> 2
- `stain_batch_001_0118.jpg`: stain_merge_dense_regions, 20 -> 6
- `stain_batch_004_0122.jpg`: stain_merge_dense_regions, 7 -> 5
- `stain_batch_004_0133.jpg`: stain_merge_dense_regions, 4 -> 2
- `stain_batch_004_0141.jpg`: stain_merge_dense_regions, 5 -> 2
- `stain_batch_004_0144.jpg`: stain_merge_dense_regions, 7 -> 5
- `stain_batch_001_0146.jpg`: stain_merge_dense_regions, 8 -> 5
- `stain_batch_004_0150.jpg`: stain_merge_dense_regions, 5 -> 3
- `stain_batch_002_0157.jpg`: stain_merge_dense_regions, 6 -> 4
- `stain_batch_004_0160.jpg`: stain_merge_dense_regions, 4 -> 2
- `crack_batch_003_0165.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_001_0168.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_001_0169.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_002_0170.jpg`: crack_shrink_tall_wide_boxes, 2 -> 2
- `crack_batch_002_0172.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_003_0173.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_001_0174.jpg`: crack_shrink_tall_wide_boxes, 4 -> 4
- `crack_batch_002_0175.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_002_0176.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1
- `crack_batch_003_0177.jpg`: crack_shrink_tall_wide_boxes, 1 -> 1

## 文件

- `optimization_manifest.csv`：所有样本的动作记录。
- `guides/`：有修改样本的 old/new 对照图。
