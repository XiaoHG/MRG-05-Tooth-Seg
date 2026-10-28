import json

import pytest
from PIL import Image

from toothseg.data import (
    export_prediction_visualizations_to_label_studio,
    export_yolo_detection_to_label_studio,
    visualize_yolo_detection_annotations,
)


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


def _write_prediction_visualization_output(root):
    (root / "01-normal").mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(root / "01-normal" / "mouth.png")
    payload = {
        "mouth.png": {
            "source_image": "01-normal/mouth.png",
            "visualization": "mouth_overlay.png",
            "image_width": 200,
            "image_height": 100,
            "detections": [
                {
                    "index": 1,
                    "class_id": 0,
                    "class_name": "tooth",
                    "confidence": 0.9,
                    "box_xyxy": [50.0, 25.0, 150.0, 75.0],
                    "box_normalized_xywh": [0.5, 0.5, 0.5, 0.5],
                }
            ],
        }
    }
    (root / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")


def test_export_prediction_visualizations_copies_sources_and_writes_predictions(tmp_path):
    predict_root = tmp_path / "predict"
    _write_prediction_visualization_output(predict_root)

    tasks_path = export_prediction_visualizations_to_label_studio(
        predict_root, tmp_path / "labelstudio" / "before"
    )

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    assert tasks[0]["id"] == "01-normal--mouth"
    assert tasks[0]["data"]["source_group"] == "01-normal"
    assert tasks[0]["data"]["image"].endswith("/01-normal--mouth.png")
    result = tasks[0]["predictions"][0]["result"][0]
    assert result["score"] == 0.9
    assert result["value"]["x"] == 25.0
    assert result["value"]["width"] == 50.0
    copied = tmp_path / "labelstudio" / "before" / "images" / "01-normal--mouth.png"
    assert copied.read_bytes() == (predict_root / "01-normal" / "mouth.png").read_bytes()
    manifest = json.loads((copied.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_count"] == 1
    assert manifest["records"][0]["box_count"] == 1


def test_export_prediction_visualizations_refuses_nonempty_output(tmp_path):
    predict_root = tmp_path / "predict"
    _write_prediction_visualization_output(predict_root)
    output = tmp_path / "before"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use overwrite=True"):
        export_prediction_visualizations_to_label_studio(predict_root, output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_visualize_yolo_annotations_preserves_sources_and_draws_boxes(tmp_path):
    dataset = tmp_path / "after"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    (dataset / "classes.txt").write_text("tooth\n", encoding="utf-8")
    image_path = dataset / "images" / "mouth.png"
    label_path = dataset / "labels" / "mouth.txt"
    Image.new("RGB", (100, 50), "white").save(image_path)
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    source_image = image_path.read_bytes()
    source_label = label_path.read_bytes()

    manifest_path = visualize_yolo_detection_annotations(dataset, dataset / "visualization")

    assert image_path.read_bytes() == source_image
    assert label_path.read_bytes() == source_label
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["image_count"] == 1
    assert manifest["box_count"] == 1
    with Image.open(dataset / "visualization" / "mouth.png") as rendered:
        assert rendered.getpixel((25, 12)) == (255, 255, 0)


def test_visualize_yolo_annotations_refuses_nonempty_output(tmp_path):
    dataset = tmp_path / "after"
    _write_raw_source(tmp_path, "after", "mouth.png", "0 0.5 0.5 0.5 0.5\n")
    output = dataset / "visualization"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use overwrite=True"):
        visualize_yolo_detection_annotations(dataset, output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
