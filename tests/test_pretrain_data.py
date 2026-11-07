import importlib.util
import io
import json
from pathlib import Path

import pytest
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


def _load_pretrain_module():
    spec = importlib.util.spec_from_file_location("pretrain", LAB_PRETRAIN / "pretrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_build_manifest_ignores_appledouble_files(tmp_path):
    module = _load_module()
    folder = tmp_path / "imgs"
    _img(folder / "a.jpg", (1, 2, 3))
    (folder / "._a.jpg").write_bytes(b"apple double metadata")

    manifest = module.build_manifest([folder], base_dir=tmp_path)
    assert manifest["seen"] == 1
    assert manifest["unique"] == 1
    assert manifest["images"][0]["path"] == "imgs/a.jpg"


def test_build_manifest_skips_corrupt_when_validation_enabled(tmp_path):
    module = _load_module()
    folder = tmp_path / "imgs"
    _img(folder / "good.png", (0, 255, 0))
    (folder / "bad.jpg").write_bytes(b"not an image")

    manifest = module.build_manifest([folder], base_dir=tmp_path, validate_images=True)
    assert manifest["seen"] == 2
    assert manifest["unique"] == 1
    assert len(manifest["corrupt"]) == 1
    assert manifest["corrupt"][0]["path"].endswith("bad.jpg")
    assert manifest["images"][0]["path"].endswith("good.png")


def test_koleo_loss_is_finite():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pretrain = _load_pretrain_module()
    torch = pretrain.torch

    z = torch.randn(8, 16, requires_grad=True)
    loss = pretrain.koLeo_loss(z)
    assert torch.isfinite(loss)
    loss.backward()  # guards against in-place ops that break autograd
    assert z.grad is not None
    # Identical features collapse to zero distance; the penalty is large but finite.
    loss_same = pretrain.koLeo_loss(torch.ones(8, 16))
    assert torch.isfinite(loss_same)


def test_run_logger_writes_metrics_and_text(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pretrain = _load_pretrain_module()

    logger = pretrain.RunLogger(
        output=tmp_path,
        log_every=1,
        effective_batch=4,
        dataset_size=100,
        max_steps=2,
        device=pretrain.torch.device("cpu"),
    )
    logger.log(0, 10.0, 1e-4, 0.996, 1.0)
    logger.log(2, 9.0, 1e-4, 0.996, 2.0)
    logger.close()

    csv_lines = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert len(csv_lines) == 3  # header + 2 rows
    assert csv_lines[0].startswith("step,epoch,loss")
    assert (tmp_path / "train.log").read_text().count("[step") == 2


def test_pretrain_dataset_handles_truncated_jpeg(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    prep = _load_module()
    folder = tmp_path / "imgs"
    folder.mkdir()

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="JPEG")
    # Drop the last 11 bytes: this is exactly the truncation that killed a
    # DataLoader worker in the server run.
    (folder / "a_truncated.jpg").write_bytes(buffer.getvalue()[:-11])
    _img(folder / "b_good.png", (0, 255, 0))

    manifest = prep.build_manifest([folder], base_dir=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pretrain = _load_pretrain_module()
    dataset = pretrain.PretrainDataset(manifest_path, base_dir=tmp_path, imgsz=32, n_local=1)
    sample = dataset[0]  # a_truncated.jpg first; must not raise
    assert len(sample["global"]) == 2
    assert len(sample["local"]) == 1
