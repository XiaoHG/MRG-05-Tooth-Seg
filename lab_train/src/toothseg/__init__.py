"""Tooth segmentation and detection helpers."""

import os

# Conda's MKL runtime and PyTorch can load separate Intel OpenMP DLLs on Windows.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

