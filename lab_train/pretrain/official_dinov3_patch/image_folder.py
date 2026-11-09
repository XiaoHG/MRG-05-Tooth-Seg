# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Minimal ImageFolder adapter for unlabeled dental pretraining data.
# Drop this file into ``dinov3/data/datasets/image_folder.py`` in the official
# repository and apply ``patch_official_dinov3.py``.

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageFolder(ExtendedVisionDataset):
    """Recursively scan ``root`` for images; labels are a constant 0.

    Self-supervised DINO training ignores the target, so a single dummy class
    is sufficient. The dataset string is ``ImageFolder:root=/path/to/images``.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Any] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._root = Path(root).resolve()
        self._paths = sorted(
            path
            for path in self._root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and not path.name.startswith("._")
        )
        if not self._paths:
            raise RuntimeError(f"No images found under {self._root}")

    def __len__(self) -> int:
        return len(self._paths)

    def get_image_relpath(self, index: int) -> str:
        return self._paths[index].relative_to(self._root).as_posix()

    def get_image_data(self, index: int) -> bytes:
        return self._paths[index].read_bytes()

    def get_target(self, index: int) -> Any:
        return 0
