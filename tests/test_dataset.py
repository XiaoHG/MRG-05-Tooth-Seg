from pathlib import Path

from PIL import Image
import pytest

from toothseg.data import prepare_raw_yolo_dataset, validate_yolo_detection_dataset


def test_dataset_is_valid():
    summary = validate_yolo_detection_dataset(Path("dataset"))
    assert summary.images == 2338
    assert summary.labels == 2338
    assert summary.classes == ["tooth"]
    assert summary.box_count > 0


def test_prepare_raw_dataset_builds_train_val_and_preserves_unlabeled_test(tmp_path):
    raw_root = tmp_path / "raw"
    for source_name in ("source-a", "source-b"):
        source_root = raw_root / source_name
        (source_root / "images").mkdir(parents=True)
        (source_root / "labels").mkdir()
        (source_root / "classes.txt").write_text("tooth\n", encoding="utf-8")
        for index in range(2):
            stem = f"{source_name}-{index}"
            Image.new("RGB", (32, 32), "white").save(source_root / "images" / f"{stem}.png")
            (source_root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    output_root = tmp_path / "dataset"
    test_dir = output_root / "images" / "test"
    test_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(test_dir / "unlabeled.png")

    data_yaml = prepare_raw_yolo_dataset(raw_root, output_root, val_ratio=0.5, seed=7)

    summary = validate_yolo_detection_dataset(output_root)
    assert data_yaml == output_root / "data.yaml"
    assert summary.images == 4
    assert summary.labels == 4
    assert (output_root / "images" / "test" / "unlabeled.png").exists()
    assert not (output_root / "labels" / "test").exists()
    assert (output_root / "dataset_manifest.json").exists()


def test_validate_rejects_duplicate_image_stems(tmp_path):
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "classes.txt").write_text("tooth\n", encoding="utf-8")
    Image.new("RGB", (32, 32), "white").save(root / "images" / "train" / "same.jpg")
    Image.new("RGB", (32, 32), "white").save(root / "images" / "train" / "same.png")
    (root / "labels" / "train" / "same.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple image files share a YOLO label stem"):
        validate_yolo_detection_dataset(root)


def test_prepare_rejects_duplicate_stems_across_raw_sources(tmp_path):
    raw_root = tmp_path / "raw"
    for source_name, suffix in (("source-a", ".jpg"), ("source-b", ".png")):
        source_root = raw_root / source_name
        (source_root / "images").mkdir(parents=True)
        (source_root / "labels").mkdir()
        (source_root / "classes.txt").write_text("tooth\n", encoding="utf-8")
        Image.new("RGB", (32, 32), "white").save(source_root / "images" / f"same{suffix}")
        (source_root / "labels" / "same.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate image stems exist across raw sources"):
        prepare_raw_yolo_dataset(raw_root, tmp_path / "dataset")
