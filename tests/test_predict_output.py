import json
from pathlib import Path

from PIL import Image
import pytest

import toothseg.model as model
from toothseg.model import save_prediction


def test_save_prediction_writes_per_image_folder(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    result = {
        "image": Image.open(image_path).convert("RGB"),
        "source": Image.open(image_path).convert("RGB"),
        "boxes": [[10.0, 12.0, 40.0, 50.0], [50.0, 10.0, 90.0, 60.0]],
        "confs": [0.9, 0.8],
    }

    out = save_prediction(result, tmp_path / "predict", image_path.stem)
    sample_dir = tmp_path / "predict" / "sample"

    assert out["dir"] == sample_dir
    assert (sample_dir / "sample_original.png").exists()
    assert (sample_dir / "sample_overlay.png").exists()
    assert (sample_dir / "sample_detections.json").exists()
    assert (sample_dir / "sample_1.png").exists()
    assert (sample_dir / "sample_2.png").exists()


def test_predict_directory_recurses_and_preserves_relative_paths(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    (image_dir / "first").mkdir(parents=True)
    (image_dir / "second").mkdir()
    for path in (image_dir / "first" / "same.png", image_dir / "second" / "other.png"):
        Image.new("RGB", (32, 24), "white").save(path)
    (image_dir / "README.txt").write_text("ignored", encoding="utf-8")

    def fake_predict_image(weights, image_path, conf=0.25):
        source = Image.open(image_path).convert("RGB")
        return {"image": source.copy(), "source": source, "boxes": [], "confs": []}

    monkeypatch.setattr(model, "predict_image", fake_predict_image)
    outputs = model.predict_directory(Path("weights.pt"), image_dir, tmp_path / "predict")

    assert len(outputs) == 2
    assert (tmp_path / "predict" / "same" / "same_overlay.png").exists()
    assert (tmp_path / "predict" / "other" / "other_overlay.png").exists()
    assert not (tmp_path / "predict" / "first").exists()
    assert not (tmp_path / "predict" / "README" / "README_overlay.png").exists()

    first_payload = (tmp_path / "predict" / "same" / "same_detections.json").read_text(encoding="utf-8")
    assert '"source_image": "first/same.png"' in first_payload
    aggregate = json.loads((tmp_path / "predict" / "predictions.json").read_text(encoding="utf-8"))
    assert aggregate["first/same.png"]["detections"] == []


def test_predict_directory_visualizations_writes_flat_overlays_and_aggregate_json(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    (image_dir / "nested").mkdir(parents=True)
    for path in (image_dir / "mouth.png", image_dir / "nested" / "molar.jpg"):
        Image.new("RGB", (32, 24), "white").save(path)

    def fake_predict_with_model(_, image_path, conf=0.25):
        source = Image.open(image_path).convert("RGB")
        return {
            "image": source.copy(),
            "source": source,
            "boxes": [[1.0, 2.0, 20.0, 18.0]],
            "confs": [0.9],
        }

    monkeypatch.setattr(model, "_require_ultralytics", lambda: lambda _: object())
    monkeypatch.setattr(model, "_predict_image_with_model", fake_predict_with_model)
    predictions_path = model.predict_directory_visualizations(
        Path("weights.pt"), image_dir, tmp_path / "predict"
    )

    assert predictions_path == tmp_path / "predict" / "predictions.json"
    assert (tmp_path / "predict" / "mouth_overlay.png").exists()
    assert (tmp_path / "predict" / "molar_overlay.png").exists()
    assert not list((tmp_path / "predict").glob("*_original.png"))
    assert not list((tmp_path / "predict").glob("*.txt"))
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert payload["mouth.png"]["source_image"] == "mouth.png"
    assert payload["mouth.png"]["image_width"] == 32
    assert payload["mouth.png"]["detections"] == [
        {
            "index": 1,
            "class_id": 0,
            "class_name": "tooth",
            "confidence": 0.9,
            "box_xyxy": [1.0, 2.0, 20.0, 18.0],
            "box_normalized_xywh": [0.328125, 0.4166666666666667, 0.59375, 0.6666666666666666],
        }
    ]
    assert payload["molar.jpg"]["source_image"] == "nested/molar.jpg"


def test_predict_directory_visualizations_rejects_duplicate_names(tmp_path):
    image_dir = tmp_path / "images"
    (image_dir / "first").mkdir(parents=True)
    (image_dir / "second").mkdir()
    Image.new("RGB", (32, 24), "white").save(image_dir / "first" / "same.png")
    Image.new("RGB", (32, 24), "white").save(image_dir / "second" / "same.png")

    with pytest.raises(ValueError, match="Duplicate image names"):
        model.predict_directory_visualizations(Path("weights.pt"), image_dir, tmp_path / "predict")


def test_predict_directory_visualizations_skips_unreadable_images(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 24), "white").save(image_dir / "good.png")
    (image_dir / "broken.jpg").write_bytes(b"not an image")

    def fake_predict_with_model(_, image_path, conf=0.25):
        if image_path.suffix == ".jpg":
            raise ValueError(f"Image could not be decoded by Ultralytics: {image_path}")
        source = Image.open(image_path).convert("RGB")
        return {"image": source.copy(), "source": source, "boxes": [], "confs": []}

    monkeypatch.setattr(model, "_require_ultralytics", lambda: lambda _: object())
    monkeypatch.setattr(model, "_predict_image_with_model", fake_predict_with_model)
    predictions_path = model.predict_directory_visualizations(
        Path("weights.pt"), image_dir, tmp_path / "predict"
    )

    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert "good.png" in payload
    assert payload["broken.jpg"]["error"]
