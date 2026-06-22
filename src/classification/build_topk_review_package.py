from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


CLASS_NAMES = {
    0: "正常",
    1: "不导电",
    2: "擦花",
    3: "横条压凹",
    4: "桔皮",
    5: "漏底",
    6: "碰伤",
    7: "起坑",
    8: "凸粉",
    9: "涂层开裂",
    10: "脏点",
}

CLASS_SOURCE_FOLDERS = {
    0: "Clean sample",
    1: "non-conducting",
    2: "scuffing",
    3: "The transverse strip is dented",
    4: "Orange peel",
    5: "Drain bottom",
    6: "Be injured by a collision",
    7: "pitting",
    8: "Convex powder",
    9: "Coating cracking",
    10: "Dirty spot",
}

CLASS_DESCRIPTIONS = {
    0: "表面整体均匀，没有明显划伤、凹坑、脏污、掉漆、开裂等异常。",
    1: "表面处理或导电性能相关异常，外观可能不明显，现场无法确认时放入不确定。",
    2: "表面有线状、片状摩擦痕、拉伤、刮擦痕，常与脏点、凸粉混淆。",
    3: "表面出现横向条纹、压痕、凹陷，通常呈横向或规则带状。",
    4: "表面纹理不均匀，类似橘皮颗粒纹，常与漏底、正常纹理混淆。",
    5: "涂层或表面覆盖不足，露出底层或出现明显颜色差异。",
    6: "撞击、磕碰、挤压造成局部破损、凹坑、变形。",
    7: "表面出现坑点、凹点、局部点状异常，常与碰伤、凸粉混淆。",
    8: "表面有凸起粉点、颗粒、小白点或颗粒状异常。",
    9: "涂层出现裂纹、断裂、开裂纹理，常与擦花或横条压凹混淆。",
    10: "黑点、污点、杂质点、污染痕迹，常与擦花、凸粉、碰伤混淆。",
}

DEFAULT_FOCUS_CLASSES = (2, 6, 8, 10)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成中文 Top1/Top3 人工复核包")
    parser.add_argument("--labeled-csv", type=Path, required=True, help="已标注图片预测结果 CSV")
    parser.add_argument("--raw-csv", type=Path, default=None, help="未分类图片预测结果 CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="中文复核包输出目录")
    parser.add_argument("--data-root", type=Path, default=None, help="训练图片根目录，用于复制 11 类示例图")
    parser.add_argument("--high-confidence", type=float, default=0.85)
    parser.add_argument("--low-confidence", type=float, default=0.60)
    parser.add_argument("--close-margin", type=float, default=0.15)
    parser.add_argument("--focus-class", action="append", type=int, default=None)
    parser.add_argument("--copy-limit-per-folder", type=int, default=0, help="0 表示复制全部命中图片")
    parser.add_argument("--no-copy", action="store_true", help="只生成索引和报告，不复制图片")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_text(value: object, max_chars: int = 80) -> str:
    text = str(value).strip()[:max_chars]
    for old in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', "\n", "\r", "\t"]:
        text = text.replace(old, "_")
    return text.replace(" ", "_")


def class_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"类别{class_id}")


def class_folder(class_id: int) -> str:
    return f"{class_id:02d}_{safe_text(class_name(class_id), 20)}"


def split_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(item) for item in str(value).split("|") if str(item).strip()]


def split_floats(value: str) -> list[float]:
    if not value:
        return []
    return [float(item) for item in str(value).split("|") if str(item).strip()]


def top_values(record: dict, raw: bool) -> tuple[int, float, list[int], list[float], float]:
    pred_id = int(record["class_id"] if raw else record["pred_class_id"])
    confidence = float(record["confidence"])
    top_ids = split_ids(str(record.get("top3_class_ids", ""))) or [pred_id]
    top_conf = split_floats(str(record.get("top3_confidence", ""))) or [confidence]
    margin = top_conf[0] - top_conf[1] if len(top_conf) >= 2 else 1.0
    return pred_id, confidence, top_ids, top_conf, margin


def decision_for(confidence: float, margin: float, high: float, low: float, close: float) -> str:
    if confidence < low:
        return "低置信度"
    if margin < close:
        return "Top1和Top2接近"
    if confidence < high:
        return "需要看Top3"
    return "直接采用Top1"


def format_top3(top_ids: Iterable[int], top_conf: Iterable[float]) -> str:
    parts = []
    for class_id, confidence in zip(top_ids, top_conf):
        parts.append(f"{class_folder(class_id)}_{confidence:.3f}")
    return "；".join(parts)


def compact_image_name(source_path: Path, prefix: str, suffix: str) -> str:
    stem = safe_text(source_path.stem, 45)
    extension = source_path.suffix.lower() or ".jpg"
    return f"{prefix}__源图_{stem}__{suffix}{extension}"


def copy_image(source_path: Path, target_dir: Path, target_name: str, counters: Counter, limit: int, no_copy: bool) -> str:
    target = target_dir / target_name
    key = str(target_dir)
    if limit > 0 and counters[key] >= limit:
        return ""
    counters[key] += 1
    if no_copy:
        return str(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return str(target)


def labeled_target_name(record: dict, pred_id: int, confidence: float, margin: float) -> str:
    source_path = Path(record["source_path"])
    true_id = int(record["true_class_id"])
    prefix = f"真实_{class_folder(true_id)}__预测_{class_folder(pred_id)}"
    suffix = f"置信度_{confidence:.3f}__前二差值_{margin:.3f}"
    return compact_image_name(source_path, prefix, suffix)


def raw_target_name(record: dict, pred_id: int, confidence: float, margin: float) -> str:
    source_path = Path(record["source_path"])
    source_name = safe_text(record.get("source_name", "未分类图片"), 20)
    prefix = f"来源_{source_name}__预测_{class_folder(pred_id)}"
    suffix = f"置信度_{confidence:.3f}__前二差值_{margin:.3f}"
    return compact_image_name(source_path, prefix, suffix)


def add_labeled_reviews(
    records: list[dict],
    output_dir: Path,
    focus_classes: set[int],
    high: float,
    low: float,
    close: float,
    limit: int,
    no_copy: bool,
) -> tuple[list[dict], Counter]:
    review_rows: list[dict] = []
    counters: Counter = Counter()
    for record in records:
        true_id = int(record["true_class_id"])
        pred_id, confidence, top_ids, top_conf, margin = top_values(record, raw=False)
        is_correct = str(record["correct"]).lower() == "true"
        decision = decision_for(confidence, margin, high=high, low=low, close=close)
        source_path = Path(record["source_path"])
        target_name = labeled_target_name(record, pred_id, confidence, margin)

        folders: list[Path] = []
        if not is_correct:
            folders.append(output_dir / "已标注图片复核" / "00_全部错分" / class_folder(true_id))
            if true_id != 0 and pred_id == 0:
                folders.append(output_dir / "已标注图片复核" / "01_缺陷判正常_优先看" / class_folder(true_id))
            if true_id == 0 and pred_id != 0:
                folders.append(output_dir / "已标注图片复核" / "02_正常判缺陷" / class_folder(pred_id))
            for class_id in focus_classes:
                if true_id == class_id or pred_id == class_id:
                    folders.append(output_dir / "已标注图片复核" / "03_四类重点错分_擦花碰伤凸粉脏点" / class_folder(class_id))
        if decision != "直接采用Top1":
            folders.append(output_dir / "Top1Top3辅助复核" / decision)

        copied_paths = [
            copy_image(source_path, folder, target_name, counters, limit=limit, no_copy=no_copy)
            for folder in folders
        ]
        if folders:
            review_rows.append(
                {
                    "数据来源": "已标注图片",
                    "原图路径": str(source_path),
                    "真实类别编号": true_id,
                    "真实类别名称": class_name(true_id),
                    "预测类别编号": pred_id,
                    "预测类别名称": class_name(pred_id),
                    "Top1置信度": f"{confidence:.6f}",
                    "Top1和Top2差值": f"{margin:.6f}",
                    "处理建议": decision,
                    "是否预测正确": "是" if is_correct else "否",
                    "Top3类别编号": "|".join(str(item) for item in top_ids),
                    "Top3说明": format_top3(top_ids, top_conf),
                    "复核目录": "|".join(str(folder) for folder in folders),
                    "复制后路径": "|".join(path for path in copied_paths if path),
                }
            )
    return review_rows, counters


def add_raw_reviews(
    records: list[dict],
    output_dir: Path,
    high: float,
    low: float,
    close: float,
    limit: int,
    no_copy: bool,
) -> tuple[list[dict], Counter]:
    review_rows: list[dict] = []
    counters: Counter = Counter()
    for record in records:
        if record.get("status", "ok") != "ok":
            continue
        pred_id, confidence, top_ids, top_conf, margin = top_values(record, raw=True)
        decision = decision_for(confidence, margin, high=high, low=low, close=close)
        source_path = Path(record["source_path"])
        target_name = raw_target_name(record, pred_id, confidence, margin)

        folders = [output_dir / "未分类图片复核" / "10_按模型预测类别分类" / class_folder(pred_id)]
        if pred_id == 0:
            folders.append(output_dir / "未分类图片复核" / "01_模型判正常_必须人工确认")
        if decision != "直接采用Top1":
            folders.append(output_dir / "未分类图片复核" / f"02_{decision}")

        copied_paths = [
            copy_image(source_path, folder, target_name, counters, limit=limit, no_copy=no_copy)
            for folder in folders
        ]
        review_rows.append(
            {
                "数据来源": "未分类图片",
                "原图路径": str(source_path),
                "真实类别编号": "",
                "真实类别名称": "",
                "预测类别编号": pred_id,
                "预测类别名称": class_name(pred_id),
                "Top1置信度": f"{confidence:.6f}",
                "Top1和Top2差值": f"{margin:.6f}",
                "处理建议": decision,
                "是否预测正确": "",
                "Top3类别编号": "|".join(str(item) for item in top_ids),
                "Top3说明": format_top3(top_ids, top_conf),
                "复核目录": "|".join(str(folder) for folder in folders),
                "复制后路径": "|".join(path for path in copied_paths if path),
            }
        )
    return review_rows, counters


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "数据来源",
        "原图路径",
        "真实类别编号",
        "真实类别名称",
        "预测类别编号",
        "预测类别名称",
        "Top1置信度",
        "Top1和Top2差值",
        "处理建议",
        "是否预测正确",
        "Top3类别编号",
        "Top3说明",
        "复核目录",
        "复制后路径",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], counters: Counter, output_md: Path, high: float, low: float, close: float) -> None:
    by_dataset = Counter(row["数据来源"] for row in rows)
    by_decision = Counter(row["处理建议"] for row in rows)
    by_pred = Counter(row["预测类别编号"] for row in rows if str(row["预测类别编号"]) != "")
    lines = [
        "# 铝型材 Top1 Top3 人工复核包汇总",
        "",
        "## 判断阈值",
        "",
        f"- 高置信度阈值：{high:.2f}",
        f"- 低置信度阈值：{low:.2f}",
        f"- Top1 和 Top2 接近阈值：{close:.2f}",
        "",
        "## 复核图片数量",
        "",
        f"- 需要复核总行数：{len(rows)}",
        f"- 已标注图片复核行数：{by_dataset.get('已标注图片', 0)}",
        f"- 未分类图片复核行数：{by_dataset.get('未分类图片', 0)}",
        "",
        "## 按处理建议统计",
        "",
        "| 处理建议 | 数量 |",
        "| --- | ---: |",
    ]
    for decision, count in by_decision.most_common():
        lines.append(f"| {decision} | {count} |")

    lines.extend(["", "## 按模型预测类别统计", "", "| 预测类别 | 数量 |", "| --- | ---: |"])
    for pred_id, count in sorted(by_pred.items(), key=lambda item: int(item[0])):
        lines.append(f"| {class_folder(int(pred_id))} | {count} |")

    lines.extend(["", "## 按输出目录统计", "", "| 输出目录 | 图片数 |", "| --- | ---: |"])
    for folder, count in sorted(counters.items()):
        lines.append(f"| `{folder}` | {count} |")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_first_image(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def copy_example_images(data_root: Path | None, doc_dir: Path, no_copy: bool) -> dict[int, Path]:
    examples: dict[int, Path] = {}
    image_dir = doc_dir / "特征图"
    for class_id, source_folder in CLASS_SOURCE_FOLDERS.items():
        if data_root is None:
            continue
        source_image = find_first_image(data_root / source_folder)
        if source_image is None:
            continue
        target = image_dir / f"{class_folder(class_id)}{source_image.suffix.lower()}"
        examples[class_id] = target
        if not no_copy:
            image_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target)
    return examples


def write_review_requirement_doc(doc_dir: Path, examples: dict[int, Path]) -> None:
    lines = [
        "# 铝型材缺陷图片人工复核要求书",
        "",
        "## 一、复核目的",
        "",
        "本次人工复核用于检查模型判断结果，重点找出漏检、误检、低置信度图片，以及未分类图片中模型判断为正常的候选图片。",
        "",
        "最重要目标：少漏检。也就是宁可多提示人工看，也不要把明显缺陷当成正常放过。",
        "",
        "## 二、图片提供与复核要求",
        "",
        "1. 请提供正常铝型材图片和不良铝型材图片。",
        "2. 图片不要加红框、红字、箭头、水印或截图标注。",
        "3. 每种常见铝型材形状都请提供一些样本。",
        "4. 尽量固定拍摄角度、距离、光源和背景。",
        "5. 不良图片中缺陷位置要清晰可见，包括擦伤、拉伤、夹渣、滴脏、破损等。",
        "6. 每种形状建议先提供正常图 800 到 1000 张、不良图 200 到 300 张。",
        "7. 如果有现场相机或工位，请尽量用未来实际检测位置拍摄，不要只用手机随手拍。",
        "8. 如果图片缺陷不明显，请在文件名中写明缺陷位置，例如：左上角擦花、右侧脏点、中间碰伤。",
        "",
        "## 三、人工复核后请这样分类",
        "",
        "请把人工确认后的图片放入下面这些中文文件夹：",
        "",
        "```text",
        "人工复核结果/",
        "  00_正常/",
        "  01_不导电/",
        "  02_擦花/",
        "  03_横条压凹/",
        "  04_桔皮/",
        "  05_漏底/",
        "  06_碰伤/",
        "  07_起坑/",
        "  08_凸粉/",
        "  09_涂层开裂/",
        "  10_脏点/",
        "  11_混合缺陷/",
        "  12_不确定/",
        "  13_坏图或无效图/",
        "  14_原标签疑似错误/",
        "```",
        "",
        "如果一张图片同时有两种或多种缺陷，请放入 `11_混合缺陷`。如果看不清、无法确定，请放入 `12_不确定`。",
        "",
        "## 四、11 种类别特征图和说明",
        "",
    ]

    for class_id in range(11):
        name = class_name(class_id)
        example = examples.get(class_id)
        lines.extend([f"### {class_id:02d}_{name}", "", CLASS_DESCRIPTIONS[class_id], ""])
        if example is not None:
            lines.extend([f"![{class_id:02d}_{name}](特征图/{example.name})", ""])
        else:
            lines.extend(["示例图：本次未找到对应图片，后续补充。", ""])

    lines.extend(
        [
            "## 五、优先复核哪些图片",
            "",
            "优先级从高到低：",
            "",
            "1. `未分类图片复核/01_模型判正常_必须人工确认`：模型认为正常，但必须人工确认。",
            "2. `已标注图片复核/01_缺陷判正常_优先看`：模型把缺陷判成正常，最容易造成漏检。",
            "3. `已标注图片复核/03_四类重点错分_擦花碰伤凸粉脏点`：擦花、碰伤、凸粉、脏点这四类容易混淆。",
            "4. `Top1Top3辅助复核/低置信度`：模型自己也不确定。",
            "5. `Top1Top3辅助复核/Top1和Top2接近`：模型第一选择和第二选择很接近，容易判断错。",
            "",
            "## 六、文件名怎么看",
            "",
            "输出图片文件名里会写明模型判断信息，例如：",
            "",
            "```text",
            "真实_02_擦花__预测_10_脏点__源图_xxx__置信度_0.642__前二差值_0.051.jpg",
            "```",
            "",
            "含义：这张图原标签是擦花，模型判断为脏点，置信度 0.642，前两名差值 0.051。差值越小，说明模型越犹豫。",
            "",
            "人工复核时不要直接相信模型结果，以图片真实内容为准。",
            "",
            "## 七、复核完成后返回什么",
            "",
            "请返回整个 `人工复核结果` 文件夹，并保留本次复核包中的 `复核索引.csv`，方便后续把人工结果回写到训练数据中。",
            "",
        ]
    )

    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "铝型材缺陷图片人工复核要求书.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    focus_classes = set(args.focus_class or DEFAULT_FOCUS_CLASSES)
    output_dir = args.output_dir
    labeled_rows = read_csv(args.labeled_csv)
    review_rows, counters = add_labeled_reviews(
        labeled_rows,
        output_dir=output_dir,
        focus_classes=focus_classes,
        high=args.high_confidence,
        low=args.low_confidence,
        close=args.close_margin,
        limit=args.copy_limit_per_folder,
        no_copy=args.no_copy,
    )

    if args.raw_csv is not None and args.raw_csv.exists():
        raw_review_rows, raw_counters = add_raw_reviews(
            read_csv(args.raw_csv),
            output_dir=output_dir,
            high=args.high_confidence,
            low=args.low_confidence,
            close=args.close_margin,
            limit=args.copy_limit_per_folder,
            no_copy=args.no_copy,
        )
        review_rows.extend(raw_review_rows)
        counters.update(raw_counters)

    doc_dir = output_dir / "说明文档"
    examples = copy_example_images(args.data_root, doc_dir, no_copy=args.no_copy)
    write_review_requirement_doc(doc_dir, examples)

    write_csv(review_rows, output_dir / "复核索引.csv")
    write_summary(
        review_rows,
        counters,
        output_md=output_dir / "汇总说明.md",
        high=args.high_confidence,
        low=args.low_confidence,
        close=args.close_margin,
    )

    print(f"复核行数={len(review_rows)}")
    print(f"输出目录={output_dir}")
    print(f"复核索引={output_dir / '复核索引.csv'}")
    print(f"汇总说明={output_dir / '汇总说明.md'}")
    print(f"人工复核要求书={doc_dir / '铝型材缺陷图片人工复核要求书.md'}")


if __name__ == "__main__":
    main()
