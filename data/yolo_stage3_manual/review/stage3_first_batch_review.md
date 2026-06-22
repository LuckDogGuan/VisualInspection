# Stage 3 First Batch Review

Review date: 2026-06-22

## Dataset

Source workspace:

```text
data/yolo_stage3_manual
```

Current classes:

```text
0 dent
1 powder
2 stain
3 crack
4 transverse_bump
```

Current split:

```text
train: 30 images
val: 10 images
test: 0 images
```

Label count:

```text
total boxes: 240
format errors: 0
```

Boxes by class:

```text
crack: 24 boxes / 8 images
dent: 50 boxes / 8 images
powder: 22 boxes / 8 images
stain: 131 boxes / 8 images
transverse_bump: 13 boxes / 8 images
```

## Fixes Applied

LabelImg wrote class ids from an old class list in some files. The box coordinates were kept, and only the first column was remapped by filename prefix:

```text
dent__ -> 0
powder__ -> 1
stain__ -> 2
crack__ -> 3
transverse_bump__ -> 4
```

The original label files were backed up to:

```text
review/original_labels_before_classid_fix
```

PowerShell wrote label files with UTF-8 BOM during the class-id fix. The labels were then rewritten as UTF-8 without BOM so Ultralytics can parse them.

## Visual Review

Annotated contact sheet:

```text
review/annotated_contact_sheet.jpg
```

The labels are usable for the first YOLO validation. Main observations:

```text
crack uses long horizontal region boxes; acceptable for rough localization.
transverse_bump uses long strip boxes; acceptable for rough localization.
stain has many tiny boxes per image; usable, but may be hard for a first detector.
dent has several images with many adjacent small boxes; later batches may benefit from merging dense adjacent damage into fewer rough boxes.
powder is generally clearer but still needs more images.
```

## Training Smoke Test

5-epoch smoke test:

```text
outputs/yolo_stage3_manual/smoke_5epoch
```

Result:

```text
The training pipeline runs successfully.
Final mAP50 was very low, as expected for only 5 epochs.
```

50-epoch small overfit test:

```text
outputs/yolo_stage3_manual/overfit_50epoch
```

Final validation result:

```text
Precision: 0.905
Recall: 0.224
mAP50: 0.300
mAP50-95: 0.155
```

Interpretation:

```text
The detector can learn from the annotations.
The dataset is too small for production use.
Recall is still low, especially for small and fragmented defects.
```

## Next Annotation Recommendation

For the next batch:

```text
Add 20 to 40 images per current class.
Keep dent, powder, stain, crack, and transverse_bump.
Do not add leak yet.
Do not add scuffing, orange peel, or non-conducting until the current five classes become more stable.
```

For marking dense small defects:

```text
If many tiny points form one local damaged area, prefer one rough region box.
If points are clearly separated, keep separate boxes.
Do not chase every barely visible speck in the first YOLO validation stage.
```

For the next training run:

```text
Use at least 30 to 50 images per class.
Use a real validation set with 8 to 10 images per class.
Run server-side training with larger image size, preferably 640.
```
