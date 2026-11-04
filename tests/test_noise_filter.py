import importlib.util
from pathlib import Path

from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "filter_noise_images", TOOLS / "filter_noise_images.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, size=(64, 64), color=(0, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_image_features_black_image(tmp_path):
    module = _load_module()
    img = tmp_path / "black.png"
    _write_image(img, color=(0, 0, 0))
    feats = module.image_features(img)
    assert feats["brightness"] == 0.0
    assert feats["highlight_ratio"] == 0.0
    # a perfectly flat image has zero Laplacian variance
    assert feats["sharpness"] == 0.0


def test_image_features_white_image(tmp_path):
    module = _load_module()
    img = tmp_path / "white.png"
    _write_image(img, color=(255, 255, 255))
    feats = module.image_features(img)
    assert feats["brightness"] == 255.0
    assert feats["highlight_ratio"] == 1.0


def test_decide_flags_all_static_reasons():
    module = _load_module()
    thr = dict(module.DEFAULT_THRESHOLDS)
    decision, reasons = module.decide(
        {"brightness": 5.0, "highlight_ratio": 0.9, "sharpness": 1.0}, None, thr
    )
    assert decision == "remove"
    assert set(reasons) == {"under_exposed", "over_exposed", "blurred"}


def test_decide_model_layer_no_teeth_and_few_low_conf():
    module = _load_module()
    thr = dict(module.DEFAULT_THRESHOLDS)
    good = {"brightness": 100.0, "highlight_ratio": 0.1, "sharpness": 100.0}

    # no teeth detected -> remove
    decision, reasons = module.decide(good, [], thr)
    assert decision == "remove"
    assert "no_teeth" in reasons

    # two teeth at low confidence -> remove
    decision, reasons = module.decide(good, [0.1, 0.2], thr)
    assert decision == "remove"
    assert "few_low_conf_teeth" in reasons

    # two teeth, one high confidence -> keep
    decision, reasons = module.decide(good, [0.1, 0.9], thr)
    assert decision == "keep"

    # many teeth -> keep
    decision, reasons = module.decide(good, [0.8] * 10, thr)
    assert decision == "keep"


def test_decide_keeps_normal_image_without_model():
    module = _load_module()
    thr = dict(module.DEFAULT_THRESHOLDS)
    decision, reasons = module.decide(
        {"brightness": 120.0, "highlight_ratio": 0.05, "sharpness": 200.0}, None, thr
    )
    assert decision == "keep"
    assert reasons == []


def test_iter_images_non_recursive_filters_suffixes(tmp_path):
    module = _load_module()
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.png")
    (tmp_path / "c.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    _write_image(tmp_path / "sub" / "d.jpg")
    found = module.iter_images(tmp_path, recursive=False)
    assert {p.name for p in found} == {"a.jpg", "b.png"}
    found_rec = module.iter_images(tmp_path, recursive=True)
    assert {p.name for p in found_rec} == {"a.jpg", "b.png", "d.jpg"}
