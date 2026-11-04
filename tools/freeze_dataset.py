"""Freeze the OMNI_COCO dataset: file list + SHA256 manifest + fingerprint.

Task v13-2 requires a frozen data list and annotation rule with a hash. This
script walks the canonical experiment data (``after`` = human-reviewed ground
truth, ``before`` = pre-review, plus ``classes.txt``/``data.yaml``), computes a
SHA256 and byte size for every file, and writes:

  <data>/freeze_manifest.json   per-file {path, size, sha256} + a single fingerprint
  <data>/FREEZE.md              human-readable freeze record + verify instructions

``--verify`` recomputes every hash and reports any mismatch against the stored
manifest. Nothing under ``dataset/`` is modified; only the two manifest files are
written next to the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "OMNI_COCO"
SCOPES = ["after", "before"]          # after = frozen ground truth (primary)
PARTS = ["images", "labels"]
SPLITS = ["train", "val", "test"]
CONFIG_FILES = ["classes.txt", "data.yaml", "data_after.yaml"]
CHUNK = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collect(data_root: Path) -> list[Path]:
    entries: list[Path] = []
    for scope in SCOPES:
        for part in PARTS:
            for split in SPLITS:
                directory = data_root / scope / part / split
                if directory.is_dir():
                    entries.extend(p for p in directory.iterdir() if p.is_file())
    for name in CONFIG_FILES:
        path = data_root / name
        if path.is_file():
            entries.append(path)
    return sorted(entries)


def _counts(data_root: Path) -> dict:
    counts: dict = {}
    for scope in SCOPES:
        counts[scope] = {}
        for part in PARTS:
            counts[scope][part] = {}
            for split in SPLITS:
                directory = data_root / scope / part / split
                counts[scope][part][split] = (
                    len([p for p in directory.iterdir() if p.is_file()]) if directory.is_dir() else 0
                )
    return counts


def _fingerprint(files: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in files:
        digest.update(f"{entry['path']}\t{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _build(data_root: Path) -> dict:
    files = []
    for path in _collect(data_root):
        files.append({
            "path": path.relative_to(data_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    files.sort(key=lambda entry: entry["path"])
    return {
        "dataset": "OMNI_COCO",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": "Frozen dataset for paper A / A0 downstream baseline.",
        "primary": "after",
        "counts": _counts(data_root),
        "files": files,
        "fingerprint": _fingerprint(files),
    }


def _write_markdown(manifest: dict, md_path: Path) -> None:
    c = manifest["counts"]
    total = len(manifest["files"])
    lines = [
        "# OMNI_COCO 数据冻结记录",
        "",
        f"- 冻结时间：`{manifest['frozen_at']}`",
        f"- 主冻结数据：`after`（人工复核后标签，作为论文 A / A0 的固定真值）",
        f"- 附带冻结：`before`（人工复核前标签，用于 before/after 对比复现）",
        f"- 文件总数：**{total}**",
        f"- 全局指纹（SHA256）：`{manifest['fingerprint']}`",
        "",
        "## 样本量",
        "",
        "| 版本 | split | images | labels |",
        "| --- | --- | --- | --- |",
    ]
    for scope in SCOPES:
        for split in SPLITS:
            lines.append(
                f"| {scope} | {split} | {c[scope]['images'][split]} | {c[scope]['labels'][split]} |"
            )
    lines += [
        "",
        "## 校验方法",
        "",
        "```powershell",
        "python tools/freeze_dataset.py --verify",
        "```",
        "",
        "任何文件被改动（增删或内容变化）都会导致校验失败。",
        "",
        "清单：`freeze_manifest.json`（逐文件 `path`/`size`/`sha256`）。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--verify", action="store_true", help="Recompute hashes and compare.")
    args = parser.parse_args()

    manifest_path = args.data / "freeze_manifest.json"
    md_path = args.data / "FREEZE.md"

    if args.verify:
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found: {manifest_path}")
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_path = {entry["path"]: entry for entry in stored["files"]}
        current = sorted(p.relative_to(args.data).as_posix() for p in _collect(args.data))

        missing = [p for p in current if p not in by_path]
        removed = [p for p in by_path if p not in current]
        changed = []
        for path in current:
            if path in by_path:
                recomputed = _sha256(args.data / path)
                if recomputed != by_path[path]["sha256"]:
                    changed.append(path)

        print(f"files in manifest: {len(by_path)}; on disk: {len(current)}")
        print(f"added: {len(missing)}; removed: {len(removed)}; changed: {len(changed)}")
        if missing:
            print("ADDED:", missing[:20])
        if removed:
            print("REMOVED:", removed[:20])
        if changed:
            print("CHANGED:", changed[:20])
        ok = not (missing or removed or changed)
        print("VERIFY:", "OK" if ok else "FAILED")
        raise SystemExit(0 if ok else 1)

    manifest = _build(args.data)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(manifest, md_path)

    print(f"Frozen {len(manifest['files'])} files -> {manifest_path}")
    print(f"fingerprint: {manifest['fingerprint']}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
