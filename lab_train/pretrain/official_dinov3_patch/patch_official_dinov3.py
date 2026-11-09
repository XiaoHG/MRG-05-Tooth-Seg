"""Patch the official facebookresearch/dinov3 repo for an ImageFolder dataset.

Usage:

    python patch_official_dinov3.py --repo /path/to/dinov3

It copies ``image_folder.py`` into ``dinov3/data/datasets/`` and adds the
``ImageFolder`` branch to ``dinov3/data/datasets/__init__.py`` and
``dinov3/data/loaders.py``. The script is idempotent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def patch_repo(repo: Path) -> None:
    target = repo / "dinov3/data/datasets/image_folder.py"
    shutil.copy2(HERE / "image_folder.py", target)
    print(f"copied: {target}")

    # Allow a backbone-only base checkpoint for ``resume_from_teacher_chkpt``:
    # heads/centers keep their initialized values when they are absent.
    checkpointer_path = repo / "dinov3/checkpointer/checkpointer.py"
    checkpointer_text = checkpointer_path.read_text(encoding="utf-8")
    old_load = """        model.load_state_dict(
            {
                key: tensor
                for key, tensor in chkpt.items()
                if not any(skip_load_key in key for skip_load_key in skip_load_keys)
            }
        )"""
    new_load = """        model.load_state_dict(
            {
                key: tensor
                for key, tensor in chkpt.items()
                if not any(skip_load_key in key for skip_load_key in skip_load_keys)
            },
            strict=False,
        )"""
    if "strict=False," not in checkpointer_text:
        if old_load not in checkpointer_text:
            raise SystemExit(f"Could not find load_state_dict anchor in {checkpointer_path}")
        checkpointer_text = checkpointer_text.replace(old_load, new_load, 1)
        checkpointer_path.write_text(checkpointer_text, encoding="utf-8")
        print(f"patched: {checkpointer_path}")
    else:
        print(f"already patched: {checkpointer_path}")

    init_path = repo / "dinov3/data/datasets/__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    import_line = "from .image_folder import ImageFolder"
    if import_line not in init_text:
        init_text = init_text.rstrip() + "\n" + import_line + "\n"
        init_path.write_text(init_text, encoding="utf-8")
        print(f"patched: {init_path}")
    else:
        print(f"already patched: {init_path}")

    loader_path = repo / "dinov3/data/loaders.py"
    loader_text = loader_path.read_text(encoding="utf-8")
    old_import = "from .datasets import ADE20K, CocoCaptions, ImageNet, ImageNet22k, NYU"
    new_import = "from .datasets import ADE20K, CocoCaptions, ImageFolder, ImageNet, ImageNet22k, NYU"
    if "ImageFolder" not in loader_text:
        if old_import not in loader_text:
            raise SystemExit(f"Unexpected import line in {loader_path}")
        loader_text = loader_text.replace(old_import, new_import)
        anchor = """    else:
        raise ValueError(f'Unsupported dataset "{name}"')"""
        replacement = """    elif name == "ImageFolder":
        class_ = ImageFolder
    else:
        raise ValueError(f'Unsupported dataset "{name}"')"""
        if anchor not in loader_text:
            raise SystemExit(f"Could not find dataset branch anchor in {loader_path}")
        loader_text = loader_text.replace(anchor, replacement, 1)
        loader_path.write_text(loader_text, encoding="utf-8")
        print(f"patched: {loader_path}")
    else:
        print(f"already patched: {loader_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / "dinov3/data/loaders.py").is_file():
        raise SystemExit(f"Not a DINOv3 repo: {repo}")
    patch_repo(repo)


if __name__ == "__main__":
    main()
