import pytest

from toothseg.model_catalog import find_model_spec, require_trainable_model


def test_model_catalog_resolves_supported_aliases():
    assert find_model_spec("yolov8m.pt").name == "yolov8m.pt"
    assert find_model_spec("yolo26m").name == "yolo26m.pt"
    assert require_trainable_model("rtdetr-l.pt").backend == "ultralytics"


def test_model_catalog_rejects_models_without_current_adapter():
    with pytest.raises(ValueError, match="adapter-needed"):
        require_trainable_model("Faster R-CNN")
    with pytest.raises(ValueError, match="RF-DETR"):
        require_trainable_model("rf-detr-m")


def test_custom_ultralytics_model_names_remain_allowed():
    spec = require_trainable_model("custom-model.pt")
    assert spec.status == "custom"
