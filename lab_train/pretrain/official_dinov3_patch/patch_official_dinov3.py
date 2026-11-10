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

    # The official formatter crashes when logger.info(cfg) receives an
    # OmegaConf DictConfig (TypeError: must be str, not DictConfig).
    logging_path = repo / "dinov3/logging/__init__.py"
    logging_text = logging_path.read_text(encoding="utf-8")
    old_msg = 'msg = record.msg % record.args if record.msg == "%s" else record.msg'
    new_msg = 'msg = record.msg % record.args if record.msg == "%s" else str(record.msg)'
    if new_msg in logging_text:
        print(f"already patched: {logging_path}")
    elif old_msg in logging_text:
        logging_text = logging_text.replace(old_msg, new_msg, 1)
        logging_path.write_text(logging_text, encoding="utf-8")
        print(f"patched: {logging_path}")
    else:
        print(f"WARNING: logging formatter anchor not found in {logging_path}")

    # Let Pillow decode mildly truncated JPEGs instead of killing the worker.
    decoders_path = repo / "dinov3/data/datasets/decoders.py"
    decoders_text = decoders_path.read_text(encoding="utf-8")
    if "LOAD_TRUNCATED_IMAGES" in decoders_text:
        print(f"already patched: {decoders_path}")
    elif "from PIL import Image\n" in decoders_text:
        decoders_text = decoders_text.replace(
            "from PIL import Image\n",
            "from PIL import Image, ImageFile\n\nImageFile.LOAD_TRUNCATED_IMAGES = True\n",
            1,
        )
        decoders_path.write_text(decoders_text, encoding="utf-8")
        print(f"patched: {decoders_path}")
    else:
        print(f"WARNING: decoder image import anchor not found in {decoders_path}")

    # Small-batch fallback: the official DINOv3 recipe hardcodes Sinkhorn-Knopp
    # centering, whose teacher target is near-uniform when the global batch is
    # tiny (e.g. 8). Allow the existing softmax+center (DINOv2-style) path.
    sslm_path = repo / "dinov3/train/ssl_meta_arch.py"
    sslm_text = sslm_path.read_text(encoding="utf-8")
    old_assert = '        assert cfg.train.centering == "sinkhorn_knopp"'
    new_assert = '        assert cfg.train.centering in ("sinkhorn_knopp", "softmax")'
    if old_assert in sslm_text:
        sslm_text = sslm_text.replace(old_assert, new_assert, 1)
    old_cls = """        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]"""
    new_cls = """        # Center with sinkhorn-knopp or softmax+center (small-batch fallback)
        if self.cfg.train.centering == "softmax":
            cls_centered = self.dino_loss.softmax_center_teacher(
                cls_after_head, teacher_temp=teacher_temp
            )
        else:
            cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
                cls_after_head, teacher_temp=teacher_temp
            )  # [n_crops * B, K]"""
    if new_cls not in sslm_text:
        if old_cls not in sslm_text:
            raise SystemExit(f"Could not find DINO centering anchor in {sslm_path}")
        sslm_text = sslm_text.replace(old_cls, new_cls, 1)
    sslm_path.write_text(sslm_text, encoding="utf-8")
    print(f"patched: {sslm_path}")

    dino_loss_path = repo / "dinov3/loss/dino_clstoken_loss.py"
    dino_loss_text = dino_loss_path.read_text(encoding="utf-8")
    old_soft = """    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)"""
    # First version of our patch updated the center with the current batch
    # before using it.  That subtracts part of the current batch from itself and
    # is noisy at batch 8.  Replace it with the DINOv2 ordering: target first,
    # center update afterwards.
    first_soft = """    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            # Per-forward center EMA for the small-batch fallback.
            batch_center = teacher_output.mean(dim=0, keepdim=True).to(self.center.dtype)
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)"""
    new_soft = """    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        # Use the center from previous steps for the current target, then update
        # it with this batch.  Updating first would subtract the current batch's
        # own mean, which is very noisy when the CLS batch has only 16 rows.
        probs = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        if update_centers:
            batch_center = teacher_output.mean(dim=0, keepdim=True).to(self.center.dtype)
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
        return probs"""
    if new_soft in dino_loss_text:
        print(f"already patched: {dino_loss_path}")
    elif first_soft in dino_loss_text:
        dino_loss_text = dino_loss_text.replace(first_soft, new_soft, 1)
        dino_loss_path.write_text(dino_loss_text, encoding="utf-8")
        print(f"re-patched (center update order): {dino_loss_path}")
    elif old_soft in dino_loss_text:
        dino_loss_text = dino_loss_text.replace(old_soft, new_soft, 1)
        dino_loss_path.write_text(dino_loss_text, encoding="utf-8")
        print(f"patched: {dino_loss_path}")
    else:
        print(f"WARNING: softmax_center_teacher anchor not found in {dino_loss_path}")

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
