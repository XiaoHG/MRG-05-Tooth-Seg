"""Convert a HuggingFace DINOv3 ViT-B/16 checkpoint to the official DINOv3 format.

The official training code (``facebookresearch/dinov3``) expects a consolidated
teacher checkpoint: ``torch.save({"teacher": state_dict}, path)``. The HF model
uses separate q/k/v projections and different key names; this script fuses and
renames them.

Usage (from ``lab_train/``):

    python pretrain/convert_hf_dinov3_to_official.py \
      --hf-model weights/dinov3-vitb16 \
      --out weights/dinov3-vitb16-official.pth \
      --official-repo /path/to/dinov3 \
      --validate-image ../dataset/positive/mild/mild_1.png

The optional validation instantiates the official ViT-B, loads the converted
state dict, and compares CLS features with the HF model on one image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def convert(hf_state: dict) -> dict:
    out = {}
    out["cls_token"] = hf_state["embeddings.cls_token"]
    out["storage_tokens"] = hf_state["embeddings.register_tokens"]
    out["mask_token"] = hf_state["embeddings.mask_token"].squeeze(1)
    out["patch_embed.proj.weight"] = hf_state["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hf_state["embeddings.patch_embeddings.bias"]

    depth = 12
    for index in range(depth):
        prefix = f"model.layer.{index}"
        block = f"blocks.{index}"
        out[f"{block}.norm1.weight"] = hf_state[f"{prefix}.norm1.weight"]
        out[f"{block}.norm1.bias"] = hf_state[f"{prefix}.norm1.bias"]

        q = hf_state[f"{prefix}.attention.q_proj.weight"]
        k = hf_state[f"{prefix}.attention.k_proj.weight"]
        v = hf_state[f"{prefix}.attention.v_proj.weight"]
        out[f"{block}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0)

        qb = hf_state[f"{prefix}.attention.q_proj.bias"]
        vb = hf_state[f"{prefix}.attention.v_proj.bias"]
        kb = torch.zeros_like(qb)
        out[f"{block}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        out[f"{block}.attn.proj.weight"] = hf_state[f"{prefix}.attention.o_proj.weight"]
        out[f"{block}.attn.proj.bias"] = hf_state[f"{prefix}.attention.o_proj.bias"]
        out[f"{block}.ls1.gamma"] = hf_state[f"{prefix}.layer_scale1.lambda1"]

        out[f"{block}.mlp.fc1.weight"] = hf_state[f"{prefix}.mlp.up_proj.weight"]
        out[f"{block}.mlp.fc1.bias"] = hf_state[f"{prefix}.mlp.up_proj.bias"]
        out[f"{block}.mlp.fc2.weight"] = hf_state[f"{prefix}.mlp.down_proj.weight"]
        out[f"{block}.mlp.fc2.bias"] = hf_state[f"{prefix}.mlp.down_proj.bias"]
        out[f"{block}.ls2.gamma"] = hf_state[f"{prefix}.layer_scale2.lambda1"]
        out[f"{block}.norm2.weight"] = hf_state[f"{prefix}.norm2.weight"]
        out[f"{block}.norm2.bias"] = hf_state[f"{prefix}.norm2.bias"]

    out["norm.weight"] = hf_state["norm.weight"]
    out["norm.bias"] = hf_state["norm.bias"]
    return out


def build_official(repo: Path, img_size: int):
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import vit_base

    return vit_base(
        patch_size=16,
        img_size=img_size,
        n_storage_tokens=4,
        qkv_bias=True,
        layerscale_init=1e-5,
        norm_layer="layernorm",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        pos_embed_type="rope",
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", type=Path, default=Path("weights/dinov3-vitb16"))
    parser.add_argument("--out", type=Path, default=Path("weights/dinov3-vitb16-official.pth"))
    parser.add_argument("--official-repo", type=Path, default=None)
    parser.add_argument("--validate-image", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    hf_model = AutoModel.from_pretrained(str(args.hf_model)).eval()
    official_state = convert(hf_model.state_dict())

    if args.official_repo is None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"teacher": official_state}, args.out)
        print(f"Saved backbone-only checkpoint: {args.out} ({len(official_state)} tensors)")
        print("WARNING: pass --official-repo to also save rope buffers and backbone.* keys.")
        return

    official = build_official(args.official_repo.resolve(), args.img_size).eval()
    message = official.load_state_dict(official_state, strict=False)
    print(f"load_state_dict: missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}")
    if message.missing_keys:
        print("missing (first 10):", message.missing_keys[:10])
    if message.unexpected_keys:
        print("unexpected (first 10):", message.unexpected_keys[:10])

    # The official training loader expects a consolidated ``teacher`` checkpoint
    # with ``backbone.`` prefixes; the model state dict also contains computed
    # buffers (e.g. rope_embed.periods) that the training loader needs.
    full_state = official.state_dict()
    training_state = {f"backbone.{key}": value for key, value in full_state.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": training_state}, args.out)
    print(f"Saved training checkpoint: {args.out} ({len(training_state)} tensors, backbone.* prefix)")

    if args.validate_image is None:
        return
    image = Image.open(args.validate_image).convert("RGB")
    scale = args.img_size / min(image.size)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.BICUBIC
    )
    left = (image.width - args.img_size) // 2
    top = (image.height - args.img_size) // 2
    image = image.crop((left, top, left + args.img_size, top + args.img_size))
    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        hf_feat = hf_model(pixel_values=tensor).last_hidden_state[:, 0]
        off_feat = official(tensor)
    cosine = F.cosine_similarity(hf_feat, off_feat, dim=-1).item()
    max_diff = (hf_feat - off_feat).abs().max().item()
    print(f"CLS cosine={cosine:.8f}  max_abs_diff={max_diff:.6g}")
    if cosine < 0.999:
        print("WARNING: conversion may be wrong (cosine < 0.999)")


if __name__ == "__main__":
    main()
