"""Record SHA256 for the comparison-model weights (v13 deliverable: 来源/版本/SHA256).

Reuses the weight paths from ``eval_test_set.MODELS``. A weight entry may be a
single file (``best.pt`` for the YOLO/torchvision models) or a directory
(``rtdetrv2-l``, whose checkpoint is ``model.safetensors`` plus metadata). For
directories every contained file is hashed.

Writes:
  output/train/omni_coco/weights_manifest.json   per-weight path + sha256 + fingerprint
  output/train/omni_coco/WEIGHTS.md               human-readable record
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from eval_test_set import MODELS  # noqa: E402

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


def _hash_weight(root: Path, rel: str) -> list[dict]:
    target = root / rel
    if not target.exists():
        return [{"path": rel, "status": "missing"}]
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.is_file())
        return [
            {"path": (p.relative_to(root).as_posix()), "size": p.stat().st_size, "sha256": _sha256(p)}
            for p in files
        ]
    return [{"path": rel, "size": target.stat().st_size, "sha256": _sha256(target)}]


def main() -> None:
    out_dir = ROOT / "output/train/omni_coco"
    entries: list[dict] = []
    for name, weights_by_variant, _ in MODELS:
        for variant in ("before", "after"):
            rel = weights_by_variant[variant]
            entries.append({"model": name, "variant": variant, "weight": rel})
            entries.extend(_hash_weight(ROOT, rel))

    fingerprint = hashlib.sha256()
    hashed = [e for e in entries if "sha256" in e]
    for e in sorted(hashed, key=lambda x: x["path"]):
        fingerprint.update(f"{e['path']}\t{e['sha256']}\n".encode("utf-8"))

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": entries,
        "fingerprint": fingerprint.hexdigest(),
    }
    (out_dir / "weights_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    missing = [e for e in entries if e.get("status") == "missing"]
    lines = [
        "# 对比模型权重 SHA256 清单",
        "",
        f"- 冻结时间：`{manifest['frozen_at']}`",
        f"- 权重指纹：`{manifest['fingerprint']}`",
        "",
        "| 模型 | 版本 | 权重路径 | 状态 |",
        "| --- | --- | --- | --- |",
    ]
    for e in entries:
        if "sha256" in e or e.get("status") == "missing":
            lines.append(
                f"| {e.get('model', '-')} | {e.get('variant', '-')} | `{e['path']}` | "
                f"{'missing' if e.get('status') == 'missing' else 'ok (' + e['sha256'][:12] + '…)'} |"
            )
    lines += [
        "",
        "完整清单：`weights_manifest.json`（逐文件 `path`/`size`/`sha256`）。",
    ]
    (out_dir / "WEIGHTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_missing = len(missing)
    print(f"weights fingerprint: {manifest['fingerprint']}")
    print(f"entries: {len(entries)}; missing: {n_missing}")
    for e in missing:
        print("MISSING:", e)
    print(f"Wrote {out_dir / 'weights_manifest.json'} and {out_dir / 'WEIGHTS.md'}")


if __name__ == "__main__":
    main()
