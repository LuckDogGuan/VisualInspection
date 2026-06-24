# 质检小缺陷判断话术模板

## 给质检工人的说明

这些是铝型材表面小缺陷候选图。请只看原图判断是否有真实缺陷，并在原图上框出位置。右侧参考图只帮助理解，不需要照抄。

## 需要标注时

```text
图片：stain_batch_001_0041.jpg
判断：需要标注
操作：merge_area
框数：3
位置说明：左侧一处，中间一处，右上两处
跳过原因：
备注：连续小点合并成稳定区域框
```

## 需要跳过时

```text
图片：stain_batch_001_0028.jpg
判断：跳过
操作：skip
框数：0
位置说明：
跳过原因：skip_unclear
备注：缺陷太淡，看不清真实边界
```

## 可用操作

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
