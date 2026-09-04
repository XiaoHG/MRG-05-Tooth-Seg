from pathlib import Path

from toothseg.data import validate_yolo_detection_dataset


def test_project_1_is_valid():
    summary = validate_yolo_detection_dataset(Path("dataset/project-1"))
    assert summary.images == 22
    assert summary.labels == 22
    assert summary.classes == ["tooth"]
    assert summary.box_count > 0

