import json

import pytest
from PIL import Image

from toothseg.data import export_yolo_detection_to_label_studio


def _write_raw_source(root, source_name, image_name, label_text):
    source = root / source_name
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    (source / "classes.txt").write_text("tooth\n", encoding="utf-8")
    Image.new("RGB", (200, 100), "white").save(source / "images" / image_name)
    (source / "labels" / f"{image_name.rsplit('.', 1)[0]}.txt").write_text(label_text, encoding="utf-8")


def test_export_to_label_studio_copies_images_and_writes_editable_predictions(tmp_path):
    raw = tmp_path / "raw"
    _write_raw_source(raw, "source-a", "mouth.png", "0 0.5 0.5 0.5 0.5\n")

    tasks_path = export_yolo_detection_to_label_studio(raw, tmp_path / "to_labelstudio")

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    assert len(tasks) == 1
    assert tasks[0]["id"] == "source-a--mouth"
    assert tasks[0]["data"]["image"] == "/data/local-files/?d=images/source-a--mouth.png"
    result = tasks[0]["predictions"][0]["result"][0]
    assert result["type"] == "rectanglelabels"
    assert result["value"] == {
        "x": 25.0,
        "y": 25.0,
        "width": 50.0,
        "height": 50.0,
        "rotation": 0,
        "rectanglelabels": ["tooth"],
    }
    assert (tmp_path / "to_labelstudio" / "images" / "source-a--mouth.png").exists()
    assert "RectangleLabels" in (tmp_path / "to_labelstudio" / "label_config.xml").read_text(encoding="utf-8")
    readme = (tmp_path / "to_labelstudio" / "README.md").read_text(encoding="utf-8")
    assert "LOCAL_FILES_DOCUMENT_ROOT" in readme
    assert "Create Project" in readme
    manifest = json.loads((tmp_path / "to_labelstudio" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_count"] == 1
    assert manifest["records"][0]["box_count"] == 1
    assert manifest["records"][0]["clipped_box_count"] == 0


def test_export_to_label_studio_clips_boxes_at_image_edges(tmp_path):
    raw = tmp_path / "raw"
    _write_raw_source(raw, "source-a", "edge.png", "0 0.96 0.5 0.1 0.4\n")

    tasks_path = export_yolo_detection_to_label_studio(raw, tmp_path / "to_labelstudio")

    task = json.loads(tasks_path.read_text(encoding="utf-8"))[0]
    value = task["predictions"][0]["result"][0]["value"]
    assert value["x"] == 91.0
    assert value["width"] == 9.0
    manifest = json.loads((tmp_path / "to_labelstudio" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"][0]["clipped_box_count"] == 1


def test_export_to_label_studio_requires_explicit_overwrite(tmp_path):
    raw = tmp_path / "raw"
    _write_raw_source(raw, "source-a", "mouth.png", "0 0.5 0.5 0.5 0.5\n")
    output = tmp_path / "to_labelstudio"
    export_yolo_detection_to_label_studio(raw, output)

    with pytest.raises(FileExistsError, match="Use overwrite=True"):
        export_yolo_detection_to_label_studio(raw, output)

    export_yolo_detection_to_label_studio(raw, output, overwrite=True)
