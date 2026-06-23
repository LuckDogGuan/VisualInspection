# YOLO Stage 3 Manual Annotation Workspace

This workspace is for the first small YOLO detection validation batch.

This first batch does not include `漏底`. The `transverse_bump` class currently uses existing `横条压凹` samples as the source for transverse strip defects. Replace or supplement it later if a separate `横条压凸` image folder becomes available.

## Class Order

Do not change `labelimg_work/classes.txt` after annotation starts. YOLO label files store class ids by line order.

```text
0 dent
1 powder
2 stain
3 crack
4 transverse_bump
```

## Workflow

1. Put selected candidate images into `candidate_images/<class>/`.
2. Copy the first review batch into `labelimg_work/images/`.
3. Open LabelImg.
4. Set image directory to `labelimg_work/images/`.
5. Set save directory to `labelimg_work/labels/`.
6. Set format to YOLO.
7. Annotate about 30 images first, then stop for review.

Do not use the old `data/yolo_dataset` as the class baseline for this stage.

Selected images and confidence values are recorded in `selection_manifest.csv`.

## Batch Tracking

All selected and annotated images are tracked in:

```text
annotation_registry.csv
```

Git stores the annotation labels, manifests, registry, scripts, and docs only.
Annotation images stay on local disk and are ignored by Git.

Rules:

```text
Do not annotate images already recorded as annotated.
Use a new folder under batches/ for each new annotation round.
Each batch has its own images/, labels/, classes.txt, predefined_classes.txt, and selection_manifest.csv.
```

Current batches:

```text
batch_001: original first 40 images, already annotated
batch_002: next 40 images, already annotated
batch_003: next 70 images, already annotated
batch_004: stain priority batch, 57 annotated images, 3 skipped images
batch_005: powder + dent batch, 60 images, waiting for annotation
```

## Next Annotation Target: Improve mAP50

The first server training run completed with overall mAP50 around `0.503`.
For the next round, optimize mAP50 first. Treat mAP50-95 as a later refinement
metric after the model can reliably find approximate defect locations.

Current class priority:

```text
P0 stain: very weak, mAP50 about 0.073
P1 powder: weak, mAP50 about 0.369
P1 dent: partial, mAP50 about 0.427
P2 transverse_bump: partial, mAP50 about 0.592
P3 crack: strong, mAP50 about 0.922
```

Recommended added labels:

```text
batch_004: stain, completed, but server_train_207 did not improve stain mAP50
batch_005: powder + dent, 30 to 40 images each
batch_006: stain review mini-batch, 20 to 30 very clear images only
batch_007: decide transverse_bump and crack hard cases after reviewing batch_005 and batch_006
```

Do not add new classes yet. Keep the class order unchanged:

```text
0 dent
1 powder
2 stain
3 crack
4 transverse_bump
```

Labeling focus:

```text
stain: label only real stain spots, not background, shadow, or reflection.
powder: box the visible powder bump or powder block, not the whole profile.
dent: box the main dent or collision mark; include clear drag marks if connected.
transverse_bump: use long boxes for real transverse strip defects.
crack: only add hard cases, such as short, thin, or low-contrast cracks.
```

Round-2 success criteria:

```text
overall mAP50 >= 0.60
stain mAP50 >= 0.30
powder mAP50 >= 0.50
dent mAP50 >= 0.50
```

Keep images out of Git. Only labels, registry, manifests, scripts, and docs are tracked.

## Batch 004 Result

`batch_004` added 57 valid stain annotations and 244 boxes. Training on
`dataset_annotated_207` completed on the server as `server_train_207`.

Result:

```text
final mAP50: 0.405
best mAP50: 0.450
stain mAP50: 0.0757
```

This means adding more stain images did not materially improve stain detection.
Before adding many more stain labels, review the stain labeling rule and keep
only very clear stain spots. Skip uncertain images.

`batch_004` is already finalized. Do not annotate it again.

For the next prepared batch, open LabelImg with the matching batch folder, for example:

```text
Open Dir:
data/yolo_stage3_manual/batches/batch_005/images

Change Save Dir:
data/yolo_stage3_manual/batches/batch_005/labels

Format:
YOLO
```

After finishing a batch, keep the labels in that batch folder and update the registry before preparing more images.

## Finalize A Finished Batch

After labeling a batch, validate it and mark it as annotated:

```text
python scripts/finalize_yolo_annotation_batch.py --batch-id batch_002
```

This checks YOLO label format, rewrites labels as UTF-8 without BOM, and updates `annotation_registry.csv`.

## Export Server Training Dataset

Only rows marked `annotated` are exported. Selected-but-unfinished batches are ignored.

```text
python scripts/build_yolo_training_dataset.py --output-name dataset_annotated_150
```

Output:

```text
data/yolo_stage3_manual/exports/dataset_annotated_150
  data.yaml
  export_manifest.csv
  images/train
  images/val
  labels/train
  labels/val
```

The exported image names are ASCII and traceable through `export_manifest.csv`.
The export script refuses to write into a non-empty export directory. Use a new
`--output-name` for each clean training dataset instead of reusing an old folder.

## Server Training

Copy the exported dataset, `yolov8n.pt`, and `scripts/train_yolo_stage3.py` to the server, then run:

```text
python scripts/train_yolo_stage3.py ^
  --data data/yolo_stage3_manual/exports/dataset_annotated_150/data.yaml ^
  --model yolov8n.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --device 0 ^
  --project outputs/yolo_stage3_manual ^
  --name server_train
```

On Linux, replace `^` with `\`.

Training output includes:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
val_batch0_labels.jpg
val_batch0_pred.jpg
training_report.md
```

Use both metrics and prediction images to decide whether training is effective.
