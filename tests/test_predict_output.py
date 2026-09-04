from pathlib import Path

from PIL import Image

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
    assert (sample_dir / "sample_overlay.png").exists()
    assert (sample_dir / "sample_detections.json").exists()
    assert (sample_dir / "sample_1.png").exists()
    assert (sample_dir / "sample_2.png").exists()
