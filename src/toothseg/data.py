from __future__ import annotations

import json
import hashlib
import math
import random
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class DatasetSummary:
    images: int
    labels: int
    classes: list[str]
    box_count: int
    min_boxes_per_image: int
    max_boxes_per_image: int


def materialize_yolo_data_yaml(dataset_root: Path, runtime_dir: Path) -> Path:
    """Create a runtime YOLO view/config from a dataset directory."""
    dataset_root = dataset_root.resolve()
    runtime_dir = runtime_dir.resolve()
    image_train = dataset_root / "images" / "train"
    split = (dataset_root / "train").is_dir() and (dataset_root / "labels" / "train").is_dir()
    standard = not split and image_train.is_dir() and (dataset_root / "labels" / "train").is_dir()
    if not standard and not split:
        raise ValueError(
            f"Unsupported dataset layout: {dataset_root}. Expected images/train + labels/train "
            "or train + labels/train."
        )

    view_root = dataset_root
    if split:
        view_root = runtime_dir / "dataset-view"
        for split_name in ("train", "val", "test"):
            image_source = dataset_root / split_name
            label_source = dataset_root / "labels" / split_name
            if image_source.is_dir() and label_source.is_dir():
                shutil.copytree(image_source, view_root / "images" / split_name, dirs_exist_ok=True)
                shutil.copytree(label_source, view_root / "labels" / split_name, dirs_exist_ok=True)

    classes_path = dataset_root / "classes.txt"
    classes = ["tooth"]
    if classes_path.is_file():
        values = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if values:
            classes = values
    runtime_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = runtime_dir / "dataset.yaml"
    lines = [
        f"path: {view_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if (view_root / "images" / "test").is_dir():
        lines.append("test: images/test")
    lines.extend(["names:", *[f"  {index}: {name}" for index, name in enumerate(classes)], ""])
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def _list_files(path: Path, suffixes: Iterable[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes])


def _list_files_recursive(path: Path, suffixes: Iterable[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def _find_duplicate_stems(paths: Iterable[Path]) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        by_stem.setdefault(path.stem, []).append(path)
    return {stem: sorted(items) for stem, items in by_stem.items() if len(items) > 1}


def _active_split_path(root: Path, kind: str) -> Path:
    split_root = root / kind
    if (split_root / "train").exists() or (split_root / "val").exists():
        return split_root
    return split_root


def load_classes(root: Path) -> list[str]:
    classes_path = root / "classes.txt"
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"Empty classes file: {classes_path}")
    return classes


def _validate_source_dataset(root: Path) -> tuple[list[Path], list[Path], list[str], int]:
    images = _list_files(root / "images", {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    labels = _list_files(root / "labels", {".txt"})
    classes = load_classes(root)
    image_by_stem = {path.stem: path for path in images}
    label_by_stem = {path.stem: path for path in labels}
    if set(image_by_stem) != set(label_by_stem):
        missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
        missing_images = sorted(set(label_by_stem) - set(image_by_stem))
        raise ValueError(
            f"Image/label mismatch in {root}. "
            f"missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
        )

    box_count = 0
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label}: {line}")
            cls, x, y, w, h = parts
            if cls != "0":
                raise ValueError(f"Unexpected class id in {label}: {cls}")
            vals = [float(x), float(y), float(w), float(h)]
            if any(v < 0 or v > 1 for v in vals):
                raise ValueError(f"Normalized coordinates out of range in {label}: {line}")
            if float(w) <= 0 or float(h) <= 0:
                raise ValueError(f"Non-positive box size in {label}: {line}")
            box_count += 1
    return images, labels, classes, box_count


def validate_yolo_detection_dataset(root: Path) -> DatasetSummary:
    images_dir = root / "images"
    labels_dir = root / "labels"
    classes = load_classes(root)

    if (images_dir / "train").exists() or (labels_dir / "train").exists():
        images = _list_files_recursive(images_dir / "train", {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) + _list_files_recursive(
            images_dir / "val", {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        labels = _list_files_recursive(labels_dir / "train", {".txt"}) + _list_files_recursive(labels_dir / "val", {".txt"})
    else:
        images = _list_files_recursive(images_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        labels = _list_files_recursive(labels_dir, {".txt"})

    duplicate_images = _find_duplicate_stems(images)
    if duplicate_images:
        details = "; ".join(
            f"{stem}: {[path.name for path in paths]}" for stem, paths in sorted(duplicate_images.items())
        )
        raise ValueError(
            "Multiple image files share a YOLO label stem; move or rename the duplicates before training: "
            + details
        )

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        raise ValueError(
            f"Image/label mismatch. missing_labels={missing_labels[:5]} missing_images={missing_images[:5]}"
        )

    box_count = 0
    per_image_counts: list[int] = []
    for label in labels:
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        per_image_counts.append(len(lines))
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label.name}: {line}")
            cls, x, y, w, h = parts
            if cls not in {"0"}:
                raise ValueError(f"Unexpected class id in {label.name}: {cls}")
            vals = [float(x), float(y), float(w), float(h)]
            if any(v < 0 or v > 1 for v in vals):
                raise ValueError(f"Normalized coordinates out of range in {label.name}: {line}")
            box_count += 1

    return DatasetSummary(
        images=len(images),
        labels=len(labels),
        classes=classes,
        box_count=box_count,
        min_boxes_per_image=min(per_image_counts) if per_image_counts else 0,
        max_boxes_per_image=max(per_image_counts) if per_image_counts else 0,
    )


def prepare_raw_yolo_dataset(
    raw_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    fixed_val_manifest: Path | None = None,
) -> Path:
    """Merge raw source directories into a reproducible train/val dataset."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    source_roots = sorted(
        path for path in raw_root.iterdir() if path.is_dir() and (path / "images").is_dir() and (path / "labels").is_dir()
    ) if raw_root.exists() else []
    if not source_roots:
        raise ValueError(f"No raw dataset sources found in: {raw_root}")

    records: list[tuple[str, Path, Path]] = []
    source_stats = []
    expected_classes: list[str] | None = None
    for source_root in source_roots:
        images, labels, classes, box_count = _validate_source_dataset(source_root)
        if expected_classes is None:
            expected_classes = classes
        elif classes != expected_classes:
            raise ValueError(f"Class definitions differ in {source_root}: {classes} != {expected_classes}")
        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}
        for stem in image_by_stem:
            records.append((source_root.name, image_by_stem[stem], label_by_stem[stem]))
        source_stats.append({"source": source_root.name, "images": len(images), "labels": len(labels), "box_count": box_count})

    assert expected_classes is not None
    names = [image.name for _, image, _ in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names exist across raw sources")
    duplicate_source_stems = _find_duplicate_stems(image for _, image, _ in records)
    if duplicate_source_stems:
        details = "; ".join(
            f"{stem}: {[f'{path.parent.parent.name}/{path.name}' for path in paths]}"
            for stem, paths in sorted(duplicate_source_stems.items())
        )
        raise ValueError(
            "Duplicate image stems exist across raw sources; resolve them before preparing the dataset: "
            + details
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            split_dir = output_root / kind / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
    for cache_name in ("train.cache", "val.cache"):
        cache_path = output_root / "labels" / cache_name
        if cache_path.exists():
            cache_path.unlink()

    fixed_val_names: set[str] = set()
    if fixed_val_manifest is not None:
        payload = json.loads(fixed_val_manifest.read_text(encoding="utf-8"))
        fixed_val_names = set(payload.get("files", {}).get("val", []))
        if not fixed_val_names:
            raise ValueError(f"Fixed validation manifest has no val files: {fixed_val_manifest}")

    rng = random.Random(seed)
    assignments: dict[str, list[tuple[str, Path, Path]]] = {"train": [], "val": []}
    for source_name in sorted({source for source, _, _ in records}):
        source_records = [record for record in records if record[0] == source_name]
        if fixed_val_manifest is not None:
            fixed = [record for record in source_records if record[1].name in fixed_val_names]
            remaining = [record for record in source_records if record[1].name not in fixed_val_names]
            assignments["val"].extend(fixed)
            assignments["train"].extend(remaining)
        else:
            rng.shuffle(source_records)
            val_count = max(1, round(len(source_records) * val_ratio))
            if val_count >= len(source_records):
                val_count = len(source_records) - 1
            assignments["val"].extend(source_records[:val_count])
            assignments["train"].extend(source_records[val_count:])

    if fixed_val_manifest is not None:
        pseudo_records = [record for record in assignments["val"] if record[0] == "pseudo-label-v6"]
        if pseudo_records:
            raise ValueError("pseudo-label-v6 contains a fixed validation filename")
        if not assignments["val"]:
            raise ValueError("Fixed validation split is empty")

    for split, split_records in assignments.items():
        for source_name, image_path, label_path in split_records:
            target_name = image_path.name
            shutil.copy2(image_path, output_root / "images" / split / target_name)
            shutil.copy2(label_path, output_root / "labels" / split / f"{image_path.stem}.txt")

    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {expected_classes[0]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "raw_root": str(raw_root),
        "output": str(output_root),
        "seed": seed,
        "val_ratio": val_ratio,
        "fixed_val_manifest": str(fixed_val_manifest) if fixed_val_manifest else None,
        "sources": source_stats,
        "train": len(assignments["train"]),
        "val": len(assignments["val"]),
        "classes": expected_classes,
    }
    (output_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_manifest = {
        **manifest,
        "files": {
            split: sorted(record[1].name for record in assignments[split])
            for split in ("train", "val")
        },
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("\n".join(expected_classes) + "\n", encoding="utf-8")
    (output_root / "notes.json").write_text(
        json.dumps(
            {
                "task": "single-class tooth detection",
                "label_format": "YOLO detection boxes",
                "raw_sources": [stat["source"] for stat in source_stats],
                "unlabeled_test_directory": "images/test",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return data_yaml


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_studio_task_id(source_name: str, image_path: Path) -> str:
    return f"{source_name}--{image_path.stem}"


def _label_studio_rectangle_result(
    task_id: str,
    index: int,
    label_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[dict[str, object], int]:
    results = []
    clipped_box_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")
        cls, x_center, y_center, width, height = parts
        if cls != "0":
            raise ValueError(f"Unexpected class id in {label_path}: {cls}")
        x_center, y_center, width, height = (float(value) for value in (x_center, y_center, width, height))
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
            raise ValueError(f"Non-finite YOLO coordinates in {label_path}: {line}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid YOLO coordinates in {label_path}: {line}")
        left = x_center - width / 2
        top = y_center - height / 2
        right = left + width
        bottom = top + height
        clipped_left = max(0.0, left)
        clipped_top = max(0.0, top)
        clipped_right = min(1.0, right)
        clipped_bottom = min(1.0, bottom)
        if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
            raise ValueError(f"YOLO box has no visible area in {label_path}: {line}")
        if (clipped_left, clipped_top, clipped_right, clipped_bottom) != (left, top, right, bottom):
            clipped_box_count += 1
        results.append(
            {
                "id": f"{task_id}-{index}-{line_number}",
                "from_name": "tooth",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "x": round(clipped_left * 100, 6),
                    "y": round(clipped_top * 100, 6),
                    "width": round((clipped_right - clipped_left) * 100, 6),
                    "height": round((clipped_bottom - clipped_top) * 100, 6),
                    "rotation": 0,
                    "rectanglelabels": ["tooth"],
                },
            }
        )
    return {"model_version": "yolo-detection-import", "result": results}, clipped_box_count


def export_yolo_detection_to_label_studio(
    raw_root: Path,
    output_root: Path,
    image_url_prefix: str = "/data/local-files/?d=images",
    overwrite: bool = False,
) -> Path:
    """Export raw YOLO detection boxes as editable Label Studio predictions."""
    source_roots = sorted(
        path for path in raw_root.iterdir() if path.is_dir() and (path / "images").is_dir() and (path / "labels").is_dir()
    ) if raw_root.exists() else []
    if not source_roots:
        raise ValueError(f"No raw dataset sources found in: {raw_root}")
    records: list[tuple[str, Path, Path]] = []
    expected_classes: list[str] | None = None
    for source_root in source_roots:
        images, labels, classes, _ = _validate_source_dataset(source_root)
        duplicate_stems = _find_duplicate_stems(images)
        if duplicate_stems:
            raise ValueError(f"Duplicate image stems in raw source {source_root}: {sorted(duplicate_stems)}")
        if expected_classes is None:
            expected_classes = classes
        elif classes != expected_classes:
            raise ValueError(f"Class definitions differ in {source_root}: {classes} != {expected_classes}")
        label_by_stem = {path.stem: path for path in labels}
        records.extend((source_root.name, image_path, label_by_stem[image_path.stem]) for image_path in images)

    image_names = [image_path.name for _, image_path, _ in records]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate image names exist across raw sources")
    duplicate_stems = _find_duplicate_stems(image_path for _, image_path, _ in records)
    if duplicate_stems:
        raise ValueError(f"Duplicate image stems exist across raw sources: {sorted(duplicate_stems)}")

    prepared_records = []
    for index, (source_name, image_path, label_path) in enumerate(records, start=1):
        task_id = _label_studio_task_id(source_name, image_path)
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image size: {image_path}")
        prediction, clipped_box_count = _label_studio_rectangle_result(
            task_id, index, label_path, image_width, image_height
        )
        prepared_records.append(
            (source_name, image_path, label_path, task_id, image_width, image_height, prediction, clipped_box_count)
        )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Label Studio output already exists: {output_root}. Use overwrite=True to replace it.")
        shutil.rmtree(output_root)

    normalized_prefix = image_url_prefix.rstrip("/")
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True)
    tasks = []
    manifest_records = []
    for source_name, image_path, label_path, task_id, _, _, prediction, clipped_box_count in prepared_records:
        target_image = images_dir / f"{task_id}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target_image)
        tasks.append(
            {
                "id": task_id,
                "data": {"image": f"{normalized_prefix}/{target_image.name}"},
                "predictions": [prediction],
            }
        )
        manifest_records.append(
            {
                "task_id": task_id,
                "image": target_image.name,
                "source": source_name,
                "source_image": str(image_path),
                "source_image_sha256": _sha256(image_path),
                "source_label": str(label_path),
                "source_label_sha256": _sha256(label_path),
                "box_count": len(prediction["result"]),
                "clipped_box_count": clipped_box_count,
                "initial_label_type": "YOLO detection boxes",
            }
        )

    label_config = """<View>\n  <Image name=\"image\" value=\"$image\"/>\n  <RectangleLabels name=\"tooth\" toName=\"image\">\n    <Label value=\"tooth\" background=\"#e74c3c\"/>\n  </RectangleLabels>\n</View>\n"""
    (output_root / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "label_config.xml").write_text(label_config, encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "v10",
                "task": "Label Studio editable tooth detection pre-annotations",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "raw_root": str(raw_root),
                "image_url_prefix": normalized_prefix,
                "classes": expected_classes,
                "task_count": len(tasks),
                "records": manifest_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        "# Label Studio 导入与配置说明\n\n"
        "本目录包含由 YOLO 牙齿检测框转换而来的可编辑矩形预标注。它不包含像素级分割掩码、FDI 牙位或氟斑牙分级；"
        "导入后的已有框是待人工修订的 predictions，不是已完成的人工真值。\n\n"
        "## 文件说明\n\n"
        "- images/：供 Label Studio 显示的图像副本，文件名含数据源前缀以保持唯一。\n"
        "- tasks.json：待导入任务及牙齿框预标注。\n"
        "- label_config.xml：单类别 tooth 矩形框标注界面配置。\n"
        "- manifest.json：任务 ID、源图像和源标签路径、SHA-256、框数及边界裁剪记录，用于复核追溯。\n\n"
        "## 导入前的本地文件配置\n\n"
        "tasks.json 中的图像地址形如：\n\n"
        "```text\n/data/local-files/?d=images/<任务图像文件名>\n```\n\n"
        "因此 Label Studio 的本地文件根目录必须设为本目录 output/to_labelstudio/，而不是其 images/ 子目录。"
        "这样地址中的 images/... 才会解析到实际图像。\n\n"
        "## 本机已验证配置\n\n"
        "本机安装的 Label Studio 1.23.0 读取的变量名是 LOCAL_FILES_SERVING_ENABLED 和 LOCAL_FILES_DOCUMENT_ROOT。"
        "不要使用 LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED 或 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT，"
        "这些变量不会被该版本读取。\n\n"
        "当前 Windows 用户环境应配置为：\n\n"
        "```text\n"
        "LOCAL_FILES_SERVING_ENABLED=true\n"
        "LOCAL_FILES_DOCUMENT_ROOT=D:\\papers\\medical-report-generation\\MRG-05-Tooth-Seg\\output\\to_labelstudio\n"
        "```\n\n"
        "配置只会被新启动的进程读取。关闭所有旧的 Label Studio 服务后，在新开的 PowerShell 中运行：\n\n"
        "```powershell\nlabel-studio start --port 8080\n```\n\n"
        "保持该窗口运行，打开 http://localhost:8080 并登录。若项目此前已导入任务，使用 Ctrl+F5 强制刷新后重新打开任务，"
        "不需要重新导入 tasks.json。\n\n"
        "## 项目 Local Files 存储关联\n\n"
        "仅设置环境变量仍不足以使任务图像可访问。Label Studio 1.23.0 会检查请求路径是否已通过 Local Files import storage "
        "关联到当前项目；未关联时服务会返回 HTTP 404，即使图像文件实际存在。\n\n"
        "在项目中打开 Settings，进入 Cloud Storage，选择 Add Source Storage，类型选择 Local Files，并填写：\n\n"
        "```text\n"
        "Absolute local path:\n"
        "D:\\papers\\medical-report-generation\\MRG-05-Tooth-Seg\\output\\to_labelstudio\\images\n"
        "```\n\n"
        "保存并验证连接后，不需要执行同步，也不要用这个存储再次导入任务；当前任务已经由 tasks.json 导入。"
        "这个存储只用于授权 Label Studio 读取任务 URL 中的 images/<文件名>。新建复核项目时，需要按同样方式重新关联一次。\n\n"
        "### 方式一：本机直接启动 Label Studio\n\n"
        "在 PowerShell 中，将下列路径替换为本目录的绝对路径，并在同一个窗口启动 Label Studio：\n\n"
        "```powershell\n"
        "$env:LOCAL_FILES_SERVING_ENABLED = \"true\"\n"
        "$env:LOCAL_FILES_DOCUMENT_ROOT = \"D:\\\\papers\\\\medical-report-generation\\\\MRG-05-Tooth-Seg\\\\output\\\\to_labelstudio\"\n"
        "label-studio start\n"
        "```\n\n"
        "首次启动后访问 http://localhost:8080，注册或登录本地账号。以上命令适合临时覆盖配置；"
        "长期使用时请按“本机已验证配置”写入 Windows 用户环境，并在新窗口中重新启动服务。\n\n"
        "### 方式二：Docker 启动 Label Studio\n\n"
        "在项目根目录的 PowerShell 中运行：\n\n"
        "```powershell\n"
        "docker run -it --rm -p 8080:8080 `\n"
        "  -v \"${PWD}\\label-studio-data:/label-studio/data\" `\n"
        "  -v \"${PWD}\\output\\to_labelstudio:/label-studio/data/to_labelstudio:ro\" `\n"
        "  -e LOCAL_FILES_SERVING_ENABLED=true `\n"
        "  -e LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/to_labelstudio `\n"
        "  heartexlabs/label-studio:latest\n"
        "```\n\n"
        "只读挂载不会让 Label Studio 修改本导出包；人工标注结果会保存在 Label Studio 自身数据目录，之后应通过界面导出。\n\n"
        "## 创建项目并导入\n\n"
        "1. 打开 Label Studio，选择 Create Project。\n"
        "2. 填写项目名称，例如 tooth-v10-review。\n"
        "3. 在 Labeling Setup 中选择 Code，用本目录的 label_config.xml 全量替换编辑器内容，然后保存项目。"
        "不要额外创建不同名称的标签；任务中的 from_name、to_name 与该配置必须一致。\n"
        "4. 在 Settings -> Cloud Storage 中添加本目录 images/ 的 Local Files import storage，步骤见“项目 Local Files 存储关联”。\n"
        "5. 进入项目后选择 Import，上传 tasks.json，等待导入完成。\n"
        "6. 打开任意任务确认图像可显示，并确认红色 tooth 矩形框已显示为预测结果。"
        "可拖动、缩放、删除或补充框后提交人工标注。\n\n"
        "## 导入后核验\n\n"
        f"导入任务数量应与 manifest.json 的 task_count 一致。本批导出为 {len(tasks)} 个任务；若数量不一致，"
        "先检查是否只导入了部分文件。对照任务 ID、图像文件名和 manifest.json，可以定位到原始数据源及标签。\n\n"
        "导出的框均已转换为百分比坐标。对于原始 YOLO 框越出图像边界的情况，导出时会裁剪到可见区域；"
        "每个任务的数量记录在 manifest.json 的 clipped_box_count。\n\n"
        "## 常见问题\n\n"
        "### 导入后图像显示为损坏、403 或 404\n\n"
        "检查本地文件服务已启用，且 LOCAL_FILES_DOCUMENT_ROOT 指向 output/to_labelstudio/。"
        "不要将根目录设为 images/，否则 URL 中的 images/ 会被重复拼接。修改环境变量后必须重启 Label Studio。\n\n"
        "在已登录状态下，任务里的相对地址会由 Label Studio 解析。不要用未登录的新浏览器窗口直接访问 /data/local-files/ "
        "判断失败，因为该接口可能返回认证错误。\n\n"
        "### 图像能显示，但看不到已有牙齿框\n\n"
        "确认项目使用的是本目录的 label_config.xml，其中 RectangleLabels 的名称为 tooth，并重新导入未被修改的 tasks.json。"
        "已有框位于 JSON 的 predictions 字段；它们需要人工审阅后才构成人工标注。\n\n"
        "### 需要重新生成或重新导入\n\n"
        "重新执行导出命令时使用 --overwrite。该操作只会重建 output/to_labelstudio/，不会修改 dataset/raw/。"
        "重新导入同一批任务前，建议新建项目或先清理旧项目，以免任务重复。\n\n"
        "## 人工复核后的数据管理\n\n"
        "从 Label Studio 导出审核结果时，保留本目录的 manifest.json 与原始任务 ID。审核完成的数据应发布到新的版本化数据源目录，"
        "例如 dataset/raw/manual-label-v10/，并在转换为 YOLO 格式后执行数据校验；不要覆盖现有 dataset/raw/ 中的历史来源。\n",
        encoding="utf-8",
    )
    return output_root / "tasks.json"


def export_pseudo_label_candidates(
    predict_root: Path,
    candidate_root: Path,
    confidence_threshold: float = 0.9,
    teacher_weights: Path | None = None,
) -> Path:
    """Export high-confidence predictions for human review without touching raw data."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    json_files = sorted(predict_root.rglob("*_detections.json"))
    if not json_files:
        raise ValueError(f"No detection JSON files found in: {predict_root}")

    images_dir = candidate_root / "images"
    labels_dir = candidate_root / "labels"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()

    records = []
    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        stem = str(payload.get("image", json_path.name.removesuffix("_detections.json")))
        source_image = json_path.parent / f"{stem}_original.png"
        if not source_image.is_file():
            raise FileNotFoundError(f"Original prediction image not found: {source_image}")
        with Image.open(source_image) as image:
            width, height = image.size
        selected = []
        for detection in payload.get("detections", []):
            confidence = float(detection.get("confidence", 0))
            if not math.isfinite(confidence) or confidence < confidence_threshold:
                continue
            box = detection.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            selected.append({"confidence": confidence, "box_xyxy": [x1, y1, x2, y2]})
        if not selected:
            continue
        candidate_image = images_dir / f"{stem}{source_image.suffix.lower()}"
        shutil.copy2(source_image, candidate_image)
        label_lines = []
        for detection in selected:
            x1, y1, x2, y2 = detection["box_xyxy"]
            label_lines.append("0 {:.8f} {:.8f} {:.8f} {:.8f}".format(
                ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                (x2 - x1) / width, (y2 - y1) / height,
            ))
        (labels_dir / f"{candidate_image.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        records.append({
            "image": candidate_image.name,
            "source_image": str(source_image),
            "source_image_sha256": _sha256(source_image),
            "source_detection_json": str(json_path),
            "candidate_boxes": selected,
            "candidate_count": len(selected),
            "review_status": "pending",
            "review_reason": None,
        })
    (candidate_root / "classes.txt").write_text("tooth\n", encoding="utf-8")
    manifest = {
        "version": "v6",
        "task": "single-class tooth detection pseudo-label candidates",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predict_root": str(predict_root),
        "teacher_weights": str(teacher_weights) if teacher_weights else None,
        "confidence_threshold": confidence_threshold,
        "review_required": True,
        "records": records,
    }
    manifest_path = candidate_root / "notes.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def publish_approved_pseudo_labels(candidate_root: Path, raw_root: Path) -> Path:
    """Publish only explicitly approved candidates into the immutable raw-source layout."""
    manifest_path = candidate_root / "notes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = raw_root / "pseudo-label-v6"
    approved = [record for record in manifest.get("records", []) if record.get("review_status") == "approved"]
    if not approved:
        raise ValueError("No approved pseudo-label records found")
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    for record in approved:
        image_name = record["image"]
        shutil.copy2(candidate_root / "images" / image_name, destination / "images" / image_name)
        shutil.copy2(candidate_root / "labels" / f"{Path(image_name).stem}.txt", destination / "labels" / f"{Path(image_name).stem}.txt")
    (destination / "classes.txt").write_text("tooth\n", encoding="utf-8")
    published = {**manifest, "published_at": datetime.now(timezone.utc).isoformat(), "published_records": len(approved)}
    (destination / "notes.json").write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def prepare_yolo_detection_dataset(
    root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    summary = validate_yolo_detection_dataset(root)
    if summary.images == 0:
        raise ValueError("Dataset is empty")

    source_root = root / "raw" if (root / "raw").exists() else root
    source_images_dir = source_root / "images"
    source_labels_dir = source_root / "labels"

    if output_root.resolve() == root.resolve():
        raw_root = root / "raw"
        raw_images = raw_root / "images"
        raw_labels = raw_root / "labels"
        raw_root.mkdir(parents=True, exist_ok=True)

        if source_root == root:
            if (root / "images").exists():
                if not raw_images.exists():
                    shutil.move(str(root / "images"), str(raw_images))
                else:
                    shutil.rmtree(root / "images", ignore_errors=True)
            if (root / "labels").exists():
                if not raw_labels.exists():
                    shutil.move(str(root / "labels"), str(raw_labels))
                else:
                    shutil.rmtree(root / "labels", ignore_errors=True)
            source_images_dir = raw_images
            source_labels_dir = raw_labels

        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            target = root / sub
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        split_root = root
    else:
        split_root = output_root
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (split_root / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    images = _list_files(source_images_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    labels_dir = source_labels_dir

    stems = [p.stem for p in images]
    rng.shuffle(stems)
    val_count = max(1, round(len(stems) * val_ratio))
    val_stems = set(stems[:val_count])
    train_stems = [stem for stem in stems if stem not in val_stems]

    if not train_stems or not val_stems:
        raise ValueError("Split produced empty train or val set")

    for stem in train_stems:
        src_img = next(p for p in images if p.stem == stem)
        src_lbl = labels_dir / f"{stem}.txt"
        shutil.copy2(src_img, split_root / "images/train" / src_img.name)
        shutil.copy2(src_lbl, split_root / "labels/train" / src_lbl.name)

    for stem in val_stems:
        src_img = next(p for p in images if p.stem == stem)
        src_lbl = labels_dir / f"{stem}.txt"
        shutil.copy2(src_img, split_root / "images/val" / src_img.name)
        shutil.copy2(src_lbl, split_root / "labels/val" / src_lbl.name)

    data_yaml = split_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {split_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {summary.classes[0]}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "source": str(root),
        "output": str(split_root),
        "source_root": str(source_root),
        "seed": seed,
        "val_ratio": val_ratio,
        "train": len(train_stems),
        "val": len(val_stems),
    }
    (split_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_yaml
