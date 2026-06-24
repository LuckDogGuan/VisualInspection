from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_yolo_annotation_review_guides import load_font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a focused stain manual review pack.")
    parser.add_argument(
        "--prediction-review-dir",
        type=Path,
        default=Path("data/yolo_stage3_manual/review/v3_prediction_review"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo_stage3_manual/review/stain_manual_review_20260624"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/yolo_stage3_manual/exports/dataset_clean_candidate_20260623_v3"),
    )
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def read_priority_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_stain_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    return [row for row in rows if row.get("class_name") == "stain"][:limit]


def suggested_decision(row: dict[str, str]) -> str:
    issue = row.get("issue", "")
    gt_count = int(row.get("gt_count") or 0)
    pred_count = int(row.get("pred_count") or 0)
    if gt_count >= 12 and pred_count <= 1:
        return "重点判断：GT 过碎。建议合并为 2-4 个稳定区域；看不清的小框删除或跳过。"
    if gt_count >= 5 and pred_count == 0:
        return "重点判断：模型完全漏检。确认缺陷是否明显；明显则合并局部区域，不明显则跳过。"
    if "重复框" in issue:
        return "重点判断：同一区域重复/过碎。合并同一区域小框，删除不明显框。"
    return "按图判断：只改明显错误，不确定保留或跳过。"


def worker_template_fields() -> list[str]:
    return [
        "priority",
        "image_name",
        "worker_judgement",
        "action_code",
        "box_count",
        "position_description",
        "skip_reason",
        "notes",
        "original_image",
        "reference_guide",
        "issue",
    ]


def chinese_worker_template_fields() -> list[str]:
    return [
        "序号",
        "图片编号",
        "判断结果",
        "处理方式",
        "框数量",
        "位置说明",
        "跳过原因",
        "备注",
        "原图文件",
        "参考图文件",
        "当前问题",
    ]


def write_decision_template(output_dir: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "priority",
        "image_name",
        "issue",
        "gt_count",
        "pred_count",
        "suggestion",
        "decision",
        "notes",
    ]
    with (output_dir / "stain_manual_decisions_template.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "priority": index,
                    "image_name": row["image_name"],
                    "issue": row["issue"],
                    "gt_count": row["gt_count"],
                    "pred_count": row["pred_count"],
                    "suggestion": suggested_decision(row),
                    "decision": "",
                    "notes": "",
                }
            )


def write_worker_reply_template(output_dir: Path, rows: list[dict[str, str]]) -> None:
    with (output_dir / "qc_worker_reply_template.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=worker_template_fields())
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "priority": index,
                    "image_name": row["image_name"],
                    "worker_judgement": "",
                    "action_code": "",
                    "box_count": "",
                    "position_description": "",
                    "skip_reason": "",
                    "notes": "",
                    "original_image": row.get("local_original", ""),
                    "reference_guide": row.get("local_guide", ""),
                    "issue": row["issue"],
                }
            )


def write_chinese_worker_reply_template(target_dir: Path, rows: list[dict[str, str]]) -> None:
    with (target_dir / "02_质检回复模板.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=chinese_worker_template_fields())
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "序号": index,
                    "图片编号": row["image_name"],
                    "判断结果": "",
                    "处理方式": "",
                    "框数量": "",
                    "位置说明": "",
                    "跳过原因": "",
                    "备注": "",
                    "原图文件": f"原图/{index:02d}_{row['image_name']}" if row.get("local_original") else "",
                    "参考图文件": f"参考图/{index:02d}_{Path(row.get('local_guide', '')).name}" if row.get("local_guide") else "",
                    "当前问题": row["issue"],
                }
            )


def write_chinese_image_list(target_dir: Path, rows: list[dict[str, str]]) -> None:
    fields = ["序号", "图片编号", "查看顺序", "原图文件", "参考图文件", "建议重点"]
    with (target_dir / "03_图片清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "序号": index,
                    "图片编号": row["image_name"],
                    "查看顺序": "先看原图，再看参考图",
                    "原图文件": f"原图/{index:02d}_{row['image_name']}" if row.get("local_original") else "",
                    "参考图文件": f"参考图/{index:02d}_{Path(row.get('local_guide', '')).name}" if row.get("local_guide") else "",
                    "建议重点": suggested_decision(row).replace("GT", "当前标注").replace("v3", "模型"),
                }
            )


def copy_guides(prediction_review_dir: Path, output_dir: Path, rows: list[dict[str, str]]) -> None:
    guide_dir = output_dir / "guides"
    guide_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        source = prediction_review_dir / row["guide_image"]
        target = guide_dir / f"{index:02d}_{source.name}"
        if source.exists():
            shutil.copy2(source, target)
            row["local_guide"] = f"guides/{target.name}"
        else:
            row["local_guide"] = ""


def find_dataset_image(dataset_root: Path, image_name: str) -> Path | None:
    for split in ("val", "train", "test"):
        candidate = dataset_root / "images" / split / image_name
        if candidate.exists():
            return candidate
    matches = list((dataset_root / "images").glob(f"*/*{image_name}")) if (dataset_root / "images").exists() else []
    return matches[0] if matches else None


def copy_originals(dataset_root: Path, output_dir: Path, rows: list[dict[str, str]]) -> None:
    original_dir = output_dir / "originals"
    original_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        source = find_dataset_image(dataset_root, row["image_name"])
        if source is None:
            row["local_original"] = ""
            continue
        target = original_dir / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        row["local_original"] = f"originals/{target.name}"


def make_contact_sheets(output_dir: Path, rows: list[dict[str, str]], per_sheet: int = 6) -> None:
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet), 1):
        chunk = rows[start : start + per_sheet]
        tiles: list[Image.Image] = []
        for offset, row in enumerate(chunk, start + 1):
            guide = output_dir / row["local_guide"]
            if not guide.exists():
                continue
            image = Image.open(guide).convert("RGB")
            image.thumbnail((720, 340), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (740, 390), "white")
            tile.paste(image, (10, 10))
            draw = ImageDraw.Draw(tile)
            draw.text((10, 356), f"{offset}. {row['image_name']} | {row['issue']}", fill=(0, 0, 0), font=font)
            tiles.append(tile)
        if not tiles:
            continue
        cols = 1
        sheet = Image.new("RGB", (740 * cols, 390 * len(tiles)), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (0, index * 390))
        output = sheet_dir / f"stain_review_sheet_{sheet_index:02d}.jpg"
        sheet.save(output, quality=92)


def make_worker_contact_sheets(output_dir: Path, rows: list[dict[str, str]], per_sheet: int = 4) -> None:
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(17)
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet), 1):
        chunk = rows[start : start + per_sheet]
        tiles: list[Image.Image] = []
        for offset, row in enumerate(chunk, start + 1):
            original_path = output_dir / row.get("local_original", "")
            guide_path = output_dir / row.get("local_guide", "")
            if not original_path.exists():
                continue
            original = Image.open(original_path).convert("RGB")
            original.thumbnail((520, 260), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (1080, 330), "white")
            tile.paste(original, (12, 34))
            draw = ImageDraw.Draw(tile)
            draw.text((12, 8), f"{offset}. 原图：{row['image_name']}", fill=(0, 0, 0), font=font)
            if guide_path.exists():
                guide = Image.open(guide_path).convert("RGB")
                guide.thumbnail((520, 260), Image.Resampling.LANCZOS)
                tile.paste(guide, (548, 34))
                draw.text((548, 8), "参考图：当前框/模型框，仅辅助判断", fill=(70, 70, 70), font=font)
            draw.text((12, 304), "请在原图上框出真实小缺陷；看不清就跳过并写原因。", fill=(0, 0, 0), font=font)
            tiles.append(tile)
        if not tiles:
            continue
        sheet = Image.new("RGB", (1080, 330 * len(tiles)), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (0, index * 330))
        output = sheet_dir / f"qc_worker_sheet_{sheet_index:02d}.jpg"
        sheet.save(output, quality=92)


def copy_worker_send_images(output_dir: Path, target_dir: Path, rows: list[dict[str, str]]) -> None:
    original_dir = target_dir / "原图"
    guide_dir = target_dir / "参考图"
    sheet_dir = target_dir / "拼图"
    original_dir.mkdir(parents=True, exist_ok=True)
    guide_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        original = output_dir / row.get("local_original", "")
        if original.exists():
            shutil.copy2(original, original_dir / f"{index:02d}_{row['image_name']}")
        guide = output_dir / row.get("local_guide", "")
        if guide.exists():
            shutil.copy2(guide, guide_dir / f"{index:02d}_{guide.name}")
    for sheet in sorted((output_dir / "contact_sheets").glob("qc_worker_sheet_*.jpg")):
        shutil.copy2(sheet, sheet_dir / sheet.name)


def write_html(output_dir: Path, rows: list[dict[str, str]]) -> None:
    body_rows = []
    for index, row in enumerate(rows, 1):
        body_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(row['image_name'])}</td>"
            f"<td>{html.escape(row['issue'])}</td>"
            f"<td>{html.escape(suggested_decision(row))}</td>"
            f"<td><img src=\"{html.escape(row['local_guide'])}\" loading=\"lazy\"></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>stain 人工判断包</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d8dee9; padding: 8px; vertical-align: top; text-align: left; }}
img {{ max-width: 1100px; width: 100%; height: auto; border: 1px solid #ccd3df; }}
code, pre {{ background: #f5f7fb; padding: 2px 4px; }}
.note {{ color: #4a5568; }}
</style>
</head>
<body>
<h1>stain 人工判断包</h1>
<p class="note">左侧绿色为当前 GT，右侧蓝色为 v3 模型预测。只处理明显问题，不确定就跳过。</p>
<h2>你只需要这样填</h2>
<pre>stain_batch_001_0041：合并成左/中/右 3 个区域；很淡的小点删除。
stain_batch_001_0028：整张不明显，跳过。
stain_batch_004_0035：GT1-GT3 合并，GT4 保留，GT5 删除。</pre>
<p>也可以直接填写 <code>stain_manual_decisions_template.csv</code> 的 decision / notes 列。</p>
<table>
<thead><tr><th>#</th><th>图片</th><th>问题</th><th>建议</th><th>对照图</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def write_worker_html(output_dir: Path, rows: list[dict[str, str]]) -> None:
    body_rows = []
    for index, row in enumerate(rows, 1):
        original_cell = (
            f"<img class=\"original\" src=\"{html.escape(row['local_original'])}\" loading=\"lazy\">"
            if row.get("local_original")
            else "<span class=\"missing\">未找到原图</span>"
        )
        guide_cell = (
            f"<img src=\"{html.escape(row['local_guide'])}\" loading=\"lazy\">"
            if row.get("local_guide")
            else "<span class=\"missing\">未找到参考图</span>"
        )
        body_rows.append(
            "<section class=\"item\">"
            f"<h2>{index}. {html.escape(row['image_name'])}</h2>"
            "<div class=\"images\">"
            f"<div><h3>原图：请以这张为准框选</h3>{original_cell}</div>"
            f"<div><h3>参考图：只帮助理解，不照抄</h3>{guide_cell}</div>"
            "</div>"
            "<div class=\"reply\">"
            "<strong>回复示例：</strong>"
            f"<code>判断=需要标注；操作=合并成 1-3 个稳定区域；位置=左侧/中间/右上；备注={html.escape(row['issue'])}</code>"
            "</div>"
            "</section>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>质检小缺陷判断包</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #172033; background: #f7f9fc; }}
h1 {{ margin-bottom: 8px; }}
.intro, .rules {{ background: white; border: 1px solid #d8dee9; padding: 14px 16px; margin-bottom: 16px; }}
.rules code, .reply code {{ background: #f0f3f8; padding: 2px 4px; }}
.item {{ background: white; border: 1px solid #d8dee9; margin: 18px 0; padding: 14px; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ccd3df; background: white; }}
.original {{ border-color: #172033; }}
h2 {{ margin: 0 0 12px; }}
h3 {{ margin: 0 0 8px; font-size: 15px; }}
li {{ margin: 4px 0; }}
.reply {{ margin-top: 10px; color: #364152; }}
.missing {{ color: #b42318; }}
@media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>质检小缺陷判断包</h1>
<div class="intro">
这些图片是 <strong>stain/脏点 小缺陷候选图</strong>。请以左侧原图为准判断，在原图上框选真实缺陷位置；右侧参考图只是帮助理解当前标注和模型结果。
</div>
<div class="rules">
<strong>统一规则</strong>
<ul>
<li>看得清、能确认是缺陷：在原图上框出缺陷主体，边缘留少量余量。</li>
<li>连续一片小缺陷：尽量合并成一个稳定区域框，不要拆太碎。</li>
<li>明显分散的多个缺陷：分开画多个框，不要用一个大框包住正常区域。</li>
<li>太淡、疑似反光、像正常纹理、看不清：跳过，并写原因。</li>
<li>不确定时写 <code>need_confirm</code>，不要强行标。</li>
</ul>
<strong>固定回复格式</strong>
<pre>图片：stain_batch_001_0041.jpg
判断：需要标注 / 跳过 / 不确定
操作：mark_one_box / mark_multiple_boxes / merge_area / delete_unclear_boxes / keep_current / skip
框数：例如 2
位置说明：例如 左侧一处，中间一处，右上两处
跳过原因：skip_unclear / skip_reflection / skip_texture / skip_not_defect / skip_duplicate / need_confirm
备注：</pre>
</div>
{''.join(body_rows)}
</body>
</html>
"""
    (output_dir / "qc_worker_index.html").write_text(document, encoding="utf-8")


def write_chinese_worker_html(target_dir: Path, rows: list[dict[str, str]]) -> None:
    body_rows = []
    for index, row in enumerate(rows, 1):
        original_name = f"原图/{index:02d}_{row['image_name']}"
        guide_name = f"参考图/{index:02d}_{Path(row.get('local_guide', '')).name}"
        body_rows.append(
            "<section class=\"item\">"
            f"<h2>{index}. {html.escape(row['image_name'])}</h2>"
            "<div class=\"images\">"
            f"<div><h3>原图：请以这张为准框选</h3><img class=\"original\" src=\"{html.escape(original_name)}\" loading=\"lazy\"></div>"
            f"<div><h3>参考图：只辅助理解，不照抄</h3><img src=\"{html.escape(guide_name)}\" loading=\"lazy\"></div>"
            "</div>"
            "<div class=\"reply\">"
            "<strong>填写建议：</strong>"
            "判断结果写“需要标注 / 跳过 / 不确定”；处理方式写“标一个框 / 标多个框 / 合并成一个区域 / 删除不明显框 / 保留当前 / 跳过”。"
            "</div>"
            "</section>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>质检小缺陷判断包</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #172033; background: #f7f9fc; }}
.intro, .rules, .item {{ background: white; border: 1px solid #d8dee9; padding: 14px 16px; margin-bottom: 16px; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ccd3df; background: white; }}
.original {{ border-color: #172033; }}
h2 {{ margin: 0 0 12px; }}
h3 {{ margin: 0 0 8px; font-size: 15px; }}
li {{ margin: 4px 0; }}
.reply {{ margin-top: 10px; color: #364152; }}
@media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>质检小缺陷判断包</h1>
<div class="intro">
这些图片是铝型材表面小缺陷候选图。请以左侧原图为准判断，并在原图上框出真实缺陷位置。右侧参考图只帮助理解，不需要照抄。
</div>
<div class="rules">
<strong>统一规则</strong>
<ul>
<li>看得清、能确认是缺陷：框出缺陷主体，边缘留少量余量。</li>
<li>连续一片小缺陷：尽量合并成一个稳定区域框，不要拆太碎。</li>
<li>明显分散的多个缺陷：分开画多个框，不要用一个大框包住正常区域。</li>
<li>太淡、疑似反光、像正常纹理、看不清：跳过，并写原因。</li>
<li>不确定：写“不确定”，不要强行标。</li>
</ul>
</div>
{''.join(body_rows)}
</body>
</html>
"""
    (target_dir / "04_查看页面.html").write_text(document, encoding="utf-8")


def write_readme(output_dir: Path, rows: list[dict[str, str]]) -> None:
    text = f"""# stain 人工判断包

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

- stain 待判断样本：{len(rows)}
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
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_worker_template_md(output_dir: Path) -> None:
    text = """# 质检小缺陷判断话术模板

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
"""
    (output_dir / "qc_worker_annotation_template.md").write_text(text, encoding="utf-8")


def write_chinese_worker_instruction(target_dir: Path, rows: list[dict[str, str]]) -> None:
    text = f"""# 质检小缺陷判断说明

## 这次需要判断什么

这批共有 {len(rows)} 张铝型材表面小缺陷候选图，主要判断是否存在“脏点、小污点、小块污染区域”等小缺陷。

请以 `原图` 文件夹里的图片为准判断。`参考图` 只是给你看当前标注和模型大致认为的位置，不需要照抄。

## 你需要返回什么

优先填写：

```text
02_质检回复模板.csv
```

如果你习惯直接在图片上画框，也可以直接在原图上框选后返回图片，同时保留图片编号。

## 怎么判断

```text
看得清、能确认是缺陷：框出缺陷主体，边缘留少量余量。
连续一片小缺陷：合并成一个稳定区域框，不要拆太碎。
明显分散的多个缺陷：分开画多个框。
太淡、疑似反光、像正常纹理、看不清：跳过，并写原因。
不确定：写“不确定”，不要强行标。
```

## 表格填写方法

`判断结果` 填下面三种之一：

```text
需要标注
跳过
不确定
```

`处理方式` 填下面一种：

```text
标一个框
标多个框
合并成一个区域
删除不明显框
保留当前
跳过
```

`跳过原因` 只在跳过或不确定时填写：

```text
缺陷太淡或看不清
疑似反光或拍摄影响
更像正常纹理或色差
不是缺陷
和其他框重复
不确定，需要确认
```

## 示例

```text
图片编号：stain_batch_001_0041.jpg
判断结果：需要标注
处理方式：合并成一个区域
框数量：3
位置说明：左侧一处，中间一处，右上两处
跳过原因：
备注：连续小点合并，不要拆太碎
```

```text
图片编号：stain_batch_001_0028.jpg
判断结果：跳过
处理方式：跳过
框数量：0
位置说明：
跳过原因：缺陷太淡或看不清
备注：看不清真实边界
```
"""
    (target_dir / "01_质检说明.md").write_text(text, encoding="utf-8")


def write_chinese_send_pack(output_dir: Path, rows: list[dict[str, str]]) -> None:
    target_dir = output_dir / "发给质检工人"
    target_dir.mkdir(parents=True, exist_ok=True)
    copy_worker_send_images(output_dir, target_dir, rows)
    write_chinese_worker_instruction(target_dir, rows)
    write_chinese_worker_reply_template(target_dir, rows)
    write_chinese_image_list(target_dir, rows)
    write_chinese_worker_html(target_dir, rows)


def build_pack(prediction_review_dir: Path, output_dir: Path, dataset_root: Path, limit: int) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_stain_rows(read_priority_rows(prediction_review_dir / "review_priority.csv"), limit=limit)
    copy_originals(dataset_root, output_dir, rows)
    copy_guides(prediction_review_dir, output_dir, rows)
    write_decision_template(output_dir, rows)
    write_worker_reply_template(output_dir, rows)
    make_contact_sheets(output_dir, rows)
    make_worker_contact_sheets(output_dir, rows)
    write_html(output_dir, rows)
    write_worker_html(output_dir, rows)
    write_readme(output_dir, rows)
    write_worker_template_md(output_dir)
    write_chinese_send_pack(output_dir, rows)
    return rows


def main() -> None:
    args = parse_args()
    rows = build_pack(args.prediction_review_dir, args.output_dir, args.dataset_root, args.limit)
    print(f"stain rows: {len(rows)}")
    print(args.output_dir / "qc_worker_index.html")
    print(args.output_dir / "qc_worker_reply_template.csv")
    print(args.output_dir / "发给质检工人")
    print(args.output_dir / "index.html")
    print(args.output_dir / "stain_manual_decisions_template.csv")


if __name__ == "__main__":
    main()
