# v3 预测复核说明

生成时间：2026-06-24

## 文件

- `index.html`：优先复核页面，左侧绿色为人工标注 GT，右侧蓝色为 v3 best 模型预测。
- `review_priority.csv`：前 40 个优先复核样本。
- `review_all.csv`：全部 53 个验证样本的复核评分。
- `guides/`：每张图片的 GT / Prediction 对照图。

## 当前判断

下一轮标注优化不要平均处理全部类别，优先顺序如下：

1. `stain`：最高优先级。当前大量漏检，并且部分 GT 被拆得太碎，模型只学到大致区域。
2. `dent`：第二优先级。主要问题是小凹坑漏检和同一缺陷重复框。
3. `powder`：第三优先级。需要确认框内是否真有明显凸粉，弱样本先不要强行加入。
4. `transverse_bump` / `crack`：这轮暂时只做明显错误修正，不作为主要标注量。

## 给人工辅助标注的规则

- 连续的 `stain` 区域：优先合并成一个稳定区域框，不要拆成很多极小框。
- 分散且明显的 `stain` 点：可以多个框，但每个框里必须能看出缺陷。
- `dent` 同一处缺陷：合并成一个框，不要对同一个凹坑画多个重叠小框。
- 框内缺陷不明显：先删或跳过，不要为了凑样本强行保留。
- `crack` 长条框：尽量贴近真实裂纹/缺陷带，避免把整条型材面都框进去。

## 优先复核样本

| 优先级 | 类别 | 问题 | 图片 |
|---:|---|---|---|
| 1 | stain | 漏检 36；多余框 1；框偏差大 | `stain_batch_001_0041.jpg` |
| 2 | stain | 漏检 21 | `stain_batch_001_0028.jpg` |
| 3 | stain | 漏检 6 | `stain_batch_004_0035.jpg` |
| 4 | stain | 漏检 6 | `stain_batch_003_0034.jpg` |
| 5 | stain | 漏检 4；多余框 2；框偏差大 | `stain_batch_004_0031.jpg` |
| 6 | stain | 漏检 5 | `stain_batch_004_0036.jpg` |
| 7 | stain | 漏检 5 | `stain_batch_004_0027.jpg` |
| 8 | stain | 漏检 5 | `stain_batch_004_0025.jpg` |
| 9 | stain | 漏检 5 | `stain_batch_003_0030.jpg` |
| 10 | dent | 漏检 3；多余框 2；框偏差大 | `dent_batch_005_0001.jpg` |
| 11 | dent | 漏检 3；重复框 1；框偏差大 | `dent_batch_005_0012.jpg` |
| 12 | stain | 漏检 4 | `stain_batch_004_0032.jpg` |
| 13 | powder | 漏检 4 | `powder_batch_001_0024.jpg` |
| 14 | stain | 漏检 3；重复框 1 | `stain_batch_004_0037.jpg` |
| 15 | powder | 漏检 3；重复框 1 | `powder_batch_002_0015.jpg` |

## 下一步

先按 `review_priority.csv` 处理前 15 到 25 张，处理完成后再生成 v4 数据集并训练。
