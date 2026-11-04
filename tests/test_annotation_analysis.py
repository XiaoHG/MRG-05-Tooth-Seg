import csv
import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_annotation", TOOLS / "analyze_annotation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_worksheet(path: Path, rows: list[dict]) -> Path:
    csv_path = path / "annotation_worksheet.csv"
    fieldnames = [
        "序号", "图像", "overlay", "子集", "GT", "预测", "TP", "FP", "FN", "错误",
        "亮度", "高光占比", "高光", "金属", "遮挡", "门牙", "其他", "备注",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows, start=1):
            full = {k: "" for k in fieldnames}
            full.update(r)
            full["序号"] = i
            writer.writerow(full)
    return csv_path


def test_flagged_parses_truthy_and_falsy():
    module = _load_module()
    assert module._flagged("1") is True
    assert module._flagged("") is False
    assert module._flagged("0") is False
    assert module._flagged(" 否 ") is False
    assert module._flagged("是") is True


def test_summarise_counts_types_and_flags_unannotated(tmp_path):
    module = _load_module()
    _write_worksheet(
        tmp_path,
        [
            {"图像": "a", "子集": "误检子集", "FP": 3, "FN": 0, "错误": 3, "高光": "1", "金属": "1"},
            {"图像": "b", "子集": "漏检子集", "FP": 0, "FN": 4, "错误": 4, "门牙": "1"},
            {"图像": "c", "子集": "整图失败", "FP": 2, "FN": 2, "错误": 4},
        ],
    )
    rows = list(csv.DictReader((tmp_path / "annotation_worksheet.csv").open(encoding="utf-8")))
    s = module._summarise(rows)

    assert s["total"] == 3
    assert len(s["annotated"]) == 2
    assert [r["图像"] for r in s["unannotated"]] == ["c"]

    assert s["per_type"]["高光"]["n"] == 1
    assert s["per_type"]["金属"]["n"] == 1
    assert s["per_type"]["门牙"]["n"] == 1
    assert s["per_type"]["遮挡"]["n"] == 0
    assert s["per_type"]["其他"]["n"] == 0

    # 高光 image has FP=3; 门牙 image has FN=4.
    assert s["per_type"]["高光"]["sum_fp"] == 3
    assert s["per_type"]["门牙"]["sum_fn"] == 4

    assert s["matrix"]["误检子集"]["高光"] == 1
    assert s["matrix"]["漏检子集"]["门牙"] == 1
    assert s["matrix"]["整图失败"]["高光"] == 0


def test_summarise_blank_sheet_is_all_unannotated(tmp_path):
    module = _load_module()
    _write_worksheet(
        tmp_path,
        [
            {"图像": "a", "子集": "误检子集", "FP": 1, "FN": 0, "错误": 1},
            {"图像": "b", "子集": "漏检子集", "FP": 0, "FN": 1, "错误": 1},
        ],
    )
    rows = list(csv.DictReader((tmp_path / "annotation_worksheet.csv").open(encoding="utf-8")))
    s = module._summarise(rows)
    assert len(s["annotated"]) == 0
    assert len(s["unannotated"]) == 2


def _load_sync_module():
    spec = importlib.util.spec_from_file_location(
        "sync_annotation_xlsx", TOOLS / "sync_annotation_xlsx.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_reads_xlsx_semantic_columns(tmp_path):
    from openpyxl import Workbook

    sync = _load_sync_module()

    wb = Workbook()
    ws = wb.active
    headers = [
        "序号", "图像", "可视化", "overlay", "子集", "GT", "预测", "TP", "FP", "FN",
        "错误", "亮度", "高光占比", "高光", "金属", "遮挡", "门牙", "其他", "备注",
    ]
    ws.append(headers)
    ws.append(["1", "img_a", "", "", "误检子集", 5, 6, 5, 1, 0, 1, 0, 0, "1", "", "", "", "", ""])
    ws.append(["2", "img_b", "", "", "漏检子集", 5, 4, 4, 0, 1, 1, 0, 0, "", "", "1", "", "", ""])
    xlsx = tmp_path / "annotation_worksheet.xlsx"
    wb.save(xlsx)

    annot = sync._read_xlsx(xlsx)
    assert annot["img_a"]["高光"] == "1"
    assert annot["img_a"]["金属"] == ""
    assert annot["img_b"]["遮挡"] == "1"
    assert annot["img_b"]["门牙"] == ""
