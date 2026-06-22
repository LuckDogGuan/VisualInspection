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
```

For `batch_002`, open LabelImg with:

```text
Open Dir:
data/yolo_stage3_manual/batches/batch_002/images

Change Save Dir:
data/yolo_stage3_manual/batches/batch_002/labels

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
