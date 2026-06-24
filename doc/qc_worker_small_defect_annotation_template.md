# 质检小缺陷判断与框选模板

适用范围：给质检工人判断铝型材表面小缺陷候选图，尤其是 `stain` 脏点、小污点、小块污染区域。

## 给质检工人的固定说明

这些图片是铝型材表面小缺陷候选图。请以原图为准判断是否有真实缺陷，并在原图上框出位置。参考图只帮助理解当前标注和模型结果，不需要照抄。

## 标注规则

```text
看得清、能确认是缺陷：框出缺陷主体，边缘留少量余量。
连续一片小缺陷：尽量合并成一个稳定区域框，不要拆太碎。
明显分散的多个缺陷：分开画多个框。
太淡、疑似反光、像正常纹理、看不清：跳过，并写原因。
不确定：写 need_confirm，不要强行标。
```

## 回复模板

```text
图片：stain_batch_001_0041.jpg
判断：需要标注 / 跳过 / 不确定
操作：mark_one_box / mark_multiple_boxes / merge_area / delete_unclear_boxes / keep_current / skip
框数：例如 2
位置说明：例如 左侧一处，中间一处，右上两处
跳过原因：skip_unclear / skip_reflection / skip_texture / skip_not_defect / skip_duplicate / need_confirm
备注：
```

## 操作代码

```text
mark_one_box：标 1 个框
mark_multiple_boxes：标多个分散框
merge_area：连续区域合并成稳定区域框
delete_unclear_boxes：删除不明显小框
keep_current：当前框可以保留
skip：跳过，不进入训练
```

## 跳过原因

```text
skip_unclear：缺陷太淡或看不清
skip_reflection：疑似反光或拍摄影响
skip_texture：更像正常纹理或色差
skip_not_defect：不是缺陷
skip_duplicate：和其他框重复
need_confirm：不确定，需要确认
```

## 示例

```text
图片：stain_batch_001_0041.jpg
判断：需要标注
操作：merge_area
框数：3
位置说明：左侧一处，中间一处，右上两处
跳过原因：
备注：连续小点合并成稳定区域框
```

```text
图片：stain_batch_001_0028.jpg
判断：跳过
操作：skip
框数：0
位置说明：
跳过原因：skip_unclear
备注：缺陷太淡，看不清真实边界
```
