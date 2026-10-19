import pytest

from toothseg.model_catalog import find_model_spec, require_trainable_model


def test_model_catalog_resolves_supported_aliases():
    assert find_model_spec("yolov8m.pt").name == "yolov8m.pt"
    assert find_model_spec("yolo26m").name == "yolo26m.pt"
    assert require_trainable_model("rtdetr-l.pt").backend == "ultralytics"


def test_model_catalog_resolves_native_adapters():
    assert require_trainable_model("Faster R-CNN").backend == "torchvision"
    assert require_trainable_model("RetinaNet").backend == "torchvision"
    assert require_trainable_model("RT-DETRv2-L").backend == "transformers"
    assert require_trainable_model("rf-detr-m").backend == "rfdetr"


def test_custom_ultralytics_model_names_remain_allowed():
    spec = require_trainable_model("custom-model.pt")
    assert spec.status == "custom"
