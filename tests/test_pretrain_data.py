import importlib.util
from pathlib import Path

from PIL import Image

LAB_PRETRAIN = Path(__file__).resolve().parents[1] / "lab_train" / "pretrain"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_pretrain_data", LAB_PRETRAIN / "prepare_pretrain_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _img(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_build_manifest_dedups_identical_images(tmp_path):
    module = _load_module()
    a = tmp_path / "a"
    b = tmp_path / "b"
    _img(a / "x.jpg", (255, 0, 0))
    _img(b / "y.jpg", (255, 0, 0))  # identical bytes -> same sha256
    _img(a / "z.png", (0, 0, 255))  # distinct

    manifest = module.build_manifest([a, b], base_dir=tmp_path)
    assert manifest["seen"] == 3
    assert manifest["unique"] == 2

    paths = {entry["path"] for entry in manifest["images"]}
    # first occurrence of the red image is kept; the duplicate is dropped
    assert "a/x.jpg" in paths
    assert "b/y.jpg" not in paths
    assert "a/z.png" in paths


def test_build_manifest_relative_paths_and_ignores_non_images(tmp_path):
    module = _load_module()
    (tmp_path / "imgs").mkdir()
    _img(tmp_path / "imgs" / "a.jpg", (1, 2, 3))
    (tmp_path / "imgs" / "note.txt").write_text("not an image")

    manifest = module.build_manifest([tmp_path / "imgs"], base_dir=tmp_path)
    assert manifest["seen"] == 1
    assert manifest["unique"] == 1
    assert manifest["images"][0]["path"] == "imgs/a.jpg"
    assert manifest["images"][0]["sha256"]
