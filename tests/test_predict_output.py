from pathlib import Path

from PIL import Image

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
