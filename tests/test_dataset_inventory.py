import importlib.util
from pathlib import Path

from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_dataset_inventory", TOOLS / "build_dataset_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_has_required_fields():
    module = _load_module()
    required = {"name", "source", "url", "license", "modality", "task", "annotation", "scale", "local_status"}
    for entry in module.DATASETS:
        missing = required - set(entry)
        assert not missing, f"{entry['name']} missing {missing}"


def test_registry_lists_known_datasets():
    module = _load_module()
    names = {e["name"] for e in module.DATASETS}
    for expected in ("OMNI_COCO", "AlphaDent", "COde-Dataset", "Oral-Diseases", "OralSDv1", "DentalAI"):
        assert expected in names


def test_scan_local_counts_images_and_labels(tmp_path):
    module = _load_module()
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "train" / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "train" / "b.png")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
    (tmp_path / "labels" / "train" / "b.txt").write_text("0 0.5 0.5 0.1 0.1")
    (tmp_path / "notes.md").write_text("x")

    scan = module._scan_local(tmp_path)
    assert scan["images"] == 2
    assert scan["labels"] == 2


def test_scan_local_missing_returns_empty():
    module = _load_module()
    scan = module._scan_local(Path("/nonexistent-dir-xyz"))
    assert scan["images"] == 0
    assert scan["labels"] == 0
