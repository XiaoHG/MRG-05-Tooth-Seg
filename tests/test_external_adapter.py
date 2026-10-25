from types import SimpleNamespace

import pytest

from toothseg import external_adapter
from toothseg import model as model_module


def test_transformers_symbols_disable_broken_optional_torchaudio(monkeypatch):
    import_utils = SimpleNamespace(_torchaudio_available=True)
    transformers = SimpleNamespace(AutoImageProcessor=object())

    def fake_import(name):
        if name == "transformers":
            return transformers
        if name == "torchaudio":
            raise OSError("DLL load failed")
        if name == "transformers.utils.import_utils":
            return import_utils
        raise ImportError(name)

    monkeypatch.setattr(external_adapter.importlib, "import_module", fake_import)
    (processor,) = external_adapter._transformers_symbols("AutoImageProcessor")

    assert processor is transformers.AutoImageProcessor
    assert import_utils._torchaudio_available is False


def test_transformers_symbols_report_missing_package(monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError("transformers")

    monkeypatch.setattr(external_adapter.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="RT-DETRv2-L requires its native package"):
        external_adapter._transformers_symbols("AutoImageProcessor")


def test_rfdetr_dispatch_omits_unspecified_learning_rate(monkeypatch, tmp_path):
    captured = {}

    def fake_train_external_detection(**kwargs):
        captured.update(kwargs)
        return tmp_path / "best.pt"

    monkeypatch.setattr(external_adapter, "train_external_detection", fake_train_external_detection)
    result = model_module.train_yolo_detection(
        tmp_path / "data.yaml", model_name="rf-detr-m", project=str(tmp_path), lr=None
    )

    assert result == tmp_path / "best.pt"
    assert "lr" not in captured
