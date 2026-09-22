import json
from pathlib import Path

from PIL import Image

from toothseg.data import export_pseudo_label_candidates, prepare_raw_yolo_dataset


def test_export_pseudo_labels_clips_boxes_and_writes_review_manifest(tmp_path):
    predictions = tmp_path / "predict"
    prediction_dir = predictions / "mouth"
    prediction_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(prediction_dir / "mouth_original.png")
    (prediction_dir / "mouth_detections.json").write_text(
        json.dumps({
            "image": "mouth",
            "detections": [
                {"confidence": 0.95, "box_xyxy": [-5, 10, 50, 90]},
                {"confidence": 0.89, "box_xyxy": [10, 10, 20, 20]},
            ],
        }),
        encoding="utf-8",
    )

    candidate_root = tmp_path / "candidates"
    manifest_path = export_pseudo_label_candidates(predictions, candidate_root)

    assert (candidate_root / "images" / "mouth.png").exists()
    assert (candidate_root / "labels" / "mouth.txt").read_text(encoding="utf-8").strip() == "0 0.25000000 0.56250000 0.50000000 0.87500000"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["records"][0]["review_status"] == "pending"
    assert len(manifest["records"][0]["candidate_boxes"]) == 1


def test_fixed_validation_manifest_keeps_pseudo_labels_in_train(tmp_path):
    raw = tmp_path / "raw"
    manifest = tmp_path / "v5-manifest.json"
    val_name = "real-val.png"
    manifest.write_text(json.dumps({"files": {"val": [val_name]} }), encoding="utf-8")
    for source, names in {
        "real": [val_name, "real-train.png"],
        "pseudo-label-v6": ["pseudo.png"],
    }.items():
        root = raw / source
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.txt").write_text("tooth\n", encoding="utf-8")
        for name in names:
            Image.new("RGB", (20, 20), "white").save(root / "images" / name)
            (root / "labels" / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    output = tmp_path / "dataset"
    prepare_raw_yolo_dataset(raw, output, fixed_val_manifest=manifest)

    assert (output / "images" / "val" / val_name).exists()
    assert (output / "images" / "train" / "pseudo.png").exists()
    assert not (output / "images" / "val" / "pseudo.png").exists()
