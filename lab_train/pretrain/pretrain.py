"""Minimal single-node DINO continue-pretraining for DINOv3 on intraoral images.

This is the A2 / v15 "OralDINO" pretraining entry point for the lab_train server.
It loads a DINOv3 backbone (HuggingFace, gated) and continues self-supervised
pretraining on the deduplicated intraoral image manifest produced by
``prepare_pretrain_data.py``.

Implemented recipe (a *minimal* DINO reference, not the full DINOv3 pipeline):
  - student/teacher EMA ViT with a DINO projection head (3-layer MLP + weight-norm
    linear over a ``out_dim`` codebook)
  - multi-crop: 2 global crops + ``local_crops`` local crops
  - DINO cross-entropy loss with teacher centering + temperature sharpening
  - AdamW + cosine LR (linear warmup) + cosine EMA momentum schedule + AMP

DINOv3's full recipe also uses iBOT / KoLeo losses and a multi-stage (gram-anchoring /
high-resolution) schedule; those are out of scope here and can be layered on later.

The model is released under the DINOv3 license and is gated on HuggingFace: set
``HF_TOKEN`` (or ``--hf-token``) after requesting access. Requires ``transformers``
with DINOv3 support and PyTorch >= 2.7.1 (already satisfied by the cu130 setup).

Usage:
  python pretrain/pretrain.py --manifest dataset/pretrain/pretrain_manifest.json \
    --model facebook/dinov3-vitb16-pretrain-lvd1689m --output output/pretrain/oraldino-vitb
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]

# Web-scraped dental datasets often contain JPEGs with a few missing bytes.
# Pillow can still decode most of them; without this flag such a file raises
# OSError and kills the whole DataLoader worker (and therefore the run).
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Resolve args with precedence CLI flags > config JSON > built-in defaults."""
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    config_path = getattr(args, "config", None)
    if config_path:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(args, key):
                setattr(args, key, value)
        # Make the config values the new defaults, then re-parse so explicit CLI wins.
        parser.set_defaults(**{k: v for k, v in vars(args).items() if v is not None})
        args = parser.parse_args(argv)
    return args


class DINOHead(nn.Module):
    """DINO projection head: Linear->GELU->Linear, l2-norm, weight-norm Linear."""

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1.0)
        self.last.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


def _make_transform(imgsz: int, global_crop: bool):
    scale = (0.4, 1.0) if global_crop else (0.05, 0.4)
    return transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PretrainDataset(Dataset):
    """Yield ``n_global`` global + ``n_local`` local crops for each manifest image."""

    def __init__(
        self,
        manifest_path: Path,
        base_dir: Path,
        imgsz: int,
        n_local: int,
        local_imgsz: int | None = None,
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Paths are stored relative to the lab_train root; resolve against the
        # root passed in (portable across machines) rather than the stored abs path.
        self.paths = [base_dir / entry["path"] for entry in manifest["images"]]
        self.global_transform = _make_transform(imgsz, global_crop=True)
        # DINO normally uses small local crops (e.g. 96/128) even when global
        # crops are large; using the full global size for locals wastes memory.
        self.local_transform = _make_transform(local_imgsz or imgsz, global_crop=False)
        self.n_local = n_local
        # Indices that failed to decode; retried only once per worker.
        self.bad_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        n = len(self.paths)
        current = index
        last_error: Exception | None = None
        for _ in range(n):
            if current in self.bad_indices:
                current = (current + 1) % n
                continue
            try:
                with Image.open(self.paths[current]) as image:
                    rgb = image.convert("RGB")
                    global_views = [self.global_transform(rgb), self.global_transform(rgb)]
                    local_views = [self.local_transform(rgb) for _ in range(self.n_local)]
                return {"global": global_views, "local": local_views}
            except (OSError, ValueError) as exc:
                self.bad_indices.add(current)
                last_error = exc
                print(
                    f"[warn] skip unreadable image: {self.paths[current]} ({exc})",
                    flush=True,
                )
                current = (current + 1) % n
        raise RuntimeError(
            f"No readable images in manifest ({n} entries). Last error: {last_error}"
        )


def _collate(batch):
    n_global = len(batch[0]["global"])
    n_local = len(batch[0]["local"])
    global_stack = torch.stack(
        [torch.stack([item["global"][i] for item in batch], dim=0) for i in range(n_global)], dim=0
    )
    local_stack = (
        torch.stack([torch.stack([item["local"][i] for item in batch], dim=0) for i in range(n_local)], dim=0)
        if n_local
        else None
    )
    return global_stack, local_stack


def dino_loss(student_out, teacher_out, center, temp_student: float, temp_teacher: float) -> torch.Tensor:
    """DINO cross-entropy between centered/sharpened teacher and log-softmax student.

    ``student_out`` is [n_views, B, K], ``teacher_out`` is [n_global, B, K].
    """
    teacher_soft = F.softmax((teacher_out - center) / temp_teacher, dim=-1).detach()
    student_logsoft = F.log_softmax(student_out / temp_student, dim=-1)

    n_global = teacher_out.shape[0]
    n_views = student_out.shape[0]
    total = torch.tensor(0.0, device=student_out.device)
    n_terms = 0
    for iq in range(n_global):
        for v in range(n_views):
            if v == iq:
                continue
            total = total + (-(teacher_soft[iq] * student_logsoft[v]).sum(dim=-1)).mean()
            n_terms += 1
    return total / max(1, n_terms)


def koLeo_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo regularizer: penalize each sample's nearest-neighbor distance.

    Encourages the batch of CLS features to spread uniformly instead of
    collapsing into a few directions. ``z`` is [B, D] (student global crops).
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=-1)
    dist = torch.cdist(z, z, p=2)
    # masked_fill is out-of-place; fill_diagonal_ would break autograd here.
    eye = torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
    min_dist = dist.masked_fill(eye, float("inf")).min(dim=1).values
    return -torch.log(min_dist + eps).mean()


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)


def cosine_schedule(base: float, final: float, step: int, total: int) -> float:
    if total <= 1:
        return final
    progress = min(1.0, step / total)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def _load_backbone(model_name: str, hf_token: str | None):
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_name, token=hf_token)


def _backbone_embed(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = backbone(pixel_values=x)
    return out.last_hidden_state[:, 0]  # CLS token


class RunLogger:
    """Write per-step metrics to CSV and a human-readable text log.

    Both files are created under ``<output>/`` by default and flushed after
    every write, so a crashed or interrupted run still leaves a usable log.
    """

    FIELDS = (
        "step",
        "epoch",
        "loss",
        "lr",
        "momentum",
        "elapsed_sec",
        "steps_per_sec",
        "samples_per_sec",
        "gpu_mem_mb",
    )

    def __init__(
        self,
        output: Path,
        log_every: int,
        effective_batch: int,
        dataset_size: int,
        max_steps: int,
        device: torch.device,
        csv_path: Path | None = None,
        text_path: Path | None = None,
    ):
        self.log_every = max(1, log_every)
        self.effective_batch = effective_batch
        self.dataset_size = max(1, dataset_size)
        self.max_steps = max_steps
        self.device = device
        self.csv_path = csv_path or (output / "metrics.csv")
        self.text_path = text_path or (output / "train.log")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._csv.flush()
        self._text = self.text_path.open("w", encoding="utf-8")

    def log(self, step: int, loss: float, lr: float, momentum: float, elapsed: float) -> None:
        if step % self.log_every != 0 and step != self.max_steps:
            return
        steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
        gpu_mem = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        row = {
            "step": step,
            "epoch": round(step * self.effective_batch / self.dataset_size, 4),
            "loss": float(loss),
            "lr": float(lr),
            "momentum": float(momentum),
            "elapsed_sec": round(elapsed, 2),
            "steps_per_sec": round(steps_per_sec, 4),
            "samples_per_sec": round(steps_per_sec * self.effective_batch, 2),
            "gpu_mem_mb": round(gpu_mem, 1),
        }
        self._writer.writerow(row)
        self._csv.flush()
        line = (
            f"[step {step}/{self.max_steps}] loss={loss:.4f} lr={lr:.2e} "
            f"momentum={momentum:.4f} epoch={row['epoch']:.2f} gpu={gpu_mem:.0f}MB"
        )
        print(line, flush=True)
        self._text.write(line + "\n")
        self._text.flush()

    def close(self) -> None:
        self._csv.close()
        self._text.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/pretrain/pretrain_manifest.json")
    parser.add_argument("--model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pretrain/oraldino-vitb")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--local-crops", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--momentum-start", type=float, default=0.996)
    parser.add_argument("--momentum-end", type=float, default=1.0)
    parser.add_argument("--temp-student", type=float, default=0.1)
    parser.add_argument("--temp-teacher", type=float, default=0.04)
    parser.add_argument("--center-ema", type=float, default=0.9)
    parser.add_argument("--koleo-weight", type=float, default=0.0,
                        help="KoLeo regularizer weight (0 disables it).")
    parser.add_argument("--out-dim", type=int, default=65536)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-student", action="store_true",
                        help="Also save the student backbone/head in checkpoints.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-imgsz", type=int, default=None,
                        help="Local crop size; defaults to --imgsz.")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Micro-batches accumulated per optimizer step.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers; use 0 for debugging or macOS.")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Write metrics every N optimizer steps.")
    parser.add_argument("--log-csv", type=Path, default=None,
                        help="CSV log path; defaults to <output>/metrics.csv.")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Text log path; defaults to <output>/train.log.")
    args = _parse_args(parser)

    if args.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.amp:
        print("AMP requested but CUDA is unavailable; disabling AMP.", flush=True)
        args.amp = False

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}. Run prepare_pretrain_data.py first.")

    dataset = PretrainDataset(
        args.manifest, ROOT, args.imgsz, args.local_crops, args.local_imgsz
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"Manifest has no images: {args.manifest}. "
            "Place images in the dataset-named folders and re-run prepare_pretrain_data.py."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        drop_last=True,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"Not enough images ({len(dataset)}) for micro-batch {args.batch} "
            "with drop_last=True; add images or lower --batch."
        )
    effective_batch = args.batch * args.grad_accum
    print(
        f"Pretraining on {len(dataset)} unique images, micro-batch {args.batch} x "
        f"grad-accum {args.grad_accum} (effective {effective_batch}), "
        f"{args.max_steps} steps, imgsz {args.imgsz} / local {args.local_imgsz or args.imgsz}, "
        f"model {args.model}"
    )

    backbone = _load_backbone(args.model, args.hf_token).to(device)
    embed_dim = backbone.config.hidden_size
    head = DINOHead(embed_dim, args.out_dim, args.hidden_dim, args.bottleneck).to(device)

    teacher_backbone = _load_backbone(args.model, args.hf_token).to(device)
    teacher_head = DINOHead(embed_dim, args.out_dim, args.hidden_dim, args.bottleneck).to(device)
    ema_update(backbone, teacher_backbone, momentum=0.0)  # hard copy at init
    ema_update(head, teacher_head, momentum=0.0)
    teacher_backbone.eval()
    teacher_head.eval()

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    center = torch.zeros(args.out_dim, device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    run_args = vars(args).copy()
    run_args["effective_batch"] = effective_batch
    run_args["dataset_size"] = len(dataset)
    (args.output / "args.json").write_text(
        json.dumps(run_args, indent=2, default=str), encoding="utf-8"
    )

    logger = RunLogger(
        output=args.output,
        log_every=args.log_every,
        effective_batch=effective_batch,
        dataset_size=len(dataset),
        max_steps=args.max_steps,
        device=device,
        csv_path=args.log_csv,
        text_path=args.log_file,
    )
    start_time = time.time()
    last_loss = float("nan")
    last_lr = args.lr
    last_momentum = args.momentum_start
    center_accum = torch.zeros_like(center)
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    backbone.train()
    head.train()
    try:
        while step < args.max_steps:
            for global_stack, local_stack in loader:
                if step >= args.max_steps:
                    break
                global_stack = global_stack.to(device)  # [2, B, 3, H, W]
                if local_stack is not None:
                    local_stack = local_stack.to(device)

                momentum = cosine_schedule(args.momentum_start, args.momentum_end, step, args.max_steps)
                lr = args.lr_min + (args.lr - args.lr_min) * cosine_schedule(1.0, 0.0, step, args.max_steps)
                if step < args.warmup_steps:
                    lr = args.lr * (step + 1) / max(1, args.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                with torch.no_grad():
                    teacher_outs = []
                    for i in range(global_stack.shape[0]):
                        teacher_outs.append(teacher_head(_backbone_embed(teacher_backbone, global_stack[i])))
                    teacher_out = torch.stack(teacher_outs, dim=0)  # [2, B, K]

                with torch.cuda.amp.autocast(enabled=args.amp):
                    student_outs = []
                    student_feats = []
                    for i in range(global_stack.shape[0]):
                        feat = _backbone_embed(backbone, global_stack[i])
                        student_feats.append(feat)
                        student_outs.append(head(feat))
                    if local_stack is not None:
                        for i in range(local_stack.shape[0]):
                            student_outs.append(head(_backbone_embed(backbone, local_stack[i])))
                    student_out = torch.stack(student_outs, dim=0)  # [2+N, B, K]
                    dino = dino_loss(student_out, teacher_out, center, args.temp_student, args.temp_teacher)
                    if args.koleo_weight > 0:
                        koleo = koLeo_loss(torch.cat(student_feats, dim=0))
                        raw_loss = dino + args.koleo_weight * koleo
                    else:
                        raw_loss = dino
                    # Scale the loss so accumulated gradients average over grad_accum micro-batches.
                    loss = raw_loss / args.grad_accum

                # Accumulate the teacher center over the optimizer's micro-batches,
                # then update it once per optimizer step (grad-accum aware).
                center_accum += teacher_out.detach().mean(dim=(0, 1))

                scaler.scale(loss).backward()
                micro += 1
                if micro % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    center = center * args.center_ema + (center_accum / args.grad_accum) * (1.0 - args.center_ema)
                    center_accum.zero_()
                    ema_update(backbone, teacher_backbone, momentum)
                    ema_update(head, teacher_head, momentum)

                    last_loss = raw_loss.item()
                    last_lr = lr
                    last_momentum = momentum
                    logger.log(step, last_loss, lr, momentum, time.time() - start_time)

                    if step > 0 and step % args.save_every == 0:
                        ckpt = args.output / f"ckpt_step{step}.pt"
                        ckpt_data = {
                            "step": step,
                            "backbone": teacher_backbone.state_dict(),
                            "head": teacher_head.state_dict(),
                            "center": center,
                        }
                        if args.save_student:
                            ckpt_data["student_backbone"] = backbone.state_dict()
                            ckpt_data["student_head"] = head.state_dict()
                        torch.save(ckpt_data, ckpt)
                        print(f"Saved checkpoint: {ckpt}", flush=True)
                    step += 1

        logger.log(step, last_loss, last_lr, last_momentum, time.time() - start_time)
    finally:
        logger.close()

    final = args.output / "teacher_checkpoint.pt"
    final_data = {
        "step": step,
        "backbone": teacher_backbone.state_dict(),
        "head": teacher_head.state_dict(),
        "center": center,
        "model": args.model,
    }
    if args.save_student:
        final_data["student_backbone"] = backbone.state_dict()
        final_data["student_head"] = head.state_dict()
    torch.save(final_data, final)
    print(f"Finished. Final teacher checkpoint: {final}")
    print(f"Metrics CSV: {logger.csv_path}")
    print(f"Text log: {logger.text_path}")


if __name__ == "__main__":
    main()
