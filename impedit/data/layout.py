"""Where the released artifacts live under IMPEDIT_DATA and IMPEDIT_WORK."""

from __future__ import annotations

import os
from pathlib import Path

from impedit.config import Paths

_BENCH_TAG = {"mmqa": "MMQA", "webqa": "WebQA"}
_INDEX_TAG = {"clip": "clip", "openclip": "laion", "laion": "laion", "siglip": "siglip"}


def mmqa_test_json(p: Paths) -> Path:
    """Official MMQA evaluation set (230 ImageQ records)."""
    return p.mmqa / "MMQA_test_image.json"


def mmqa_metadata_json(p: Paths) -> Path:
    """Upstream doc_id -> image-path table used to locate all 228 KB images."""
    return p.mmqa / "MMQA_image_metadata.json"


def mmqa_seed_targets_json(p: Paths) -> Path:
    """The original 50 targets; input of the 100-target expansion."""
    return p.mmqa / "MMQA-targets-seed.json"


def mmqa_targets_json(p: Paths) -> Path:
    """100 targets with designed wrong answers (targets.py mmqa-targets)."""
    return p.mmqa / "MMQA-targets-100.json"


def mmqa_index_json(p: Paths) -> Path:
    """Unified index: every evaluation query tagged target or benign."""
    override = os.environ.get("MMQA_INDEX")
    return Path(override) if override else p.mmqa / "MMQA_ImageQ_index.json"


def mmqa_images(p: Paths) -> Path:
    return p.mmqa / "images" / "final_dataset_images"


def webqa_test_json(p: Paths) -> Path:
    """Official WebQA evaluation set; a dict keyed by qid."""
    return p.webqa / "WebQA_test_image.json"


def webqa_targets_json(p: Paths) -> Path:
    """100 WebQA targets; the payload loop writes image paths back into it."""
    return p.webqa / "WebQA-targets.json"


def webqa_images(p: Paths) -> Path:
    return p.webqa / "images"


def target_manifest(p: Paths) -> Path:
    return p.targets / "target_manifest.json"


def index_to_id(p: Paths, bench: str) -> Path:
    """``<TASK>_test_image_index_to_id.json`` (slot -> document id), from IMPEDIT_DATA."""
    tag = _BENCH_TAG[bench]
    base = p.mmqa if bench == "mmqa" else p.webqa
    for name in (f"{tag}_test_image_index_to_id.json", f"{tag.upper()}_test_image_index_to_id.json"):
        f = base / name
        if f.exists():
            return f
    return base / f"{tag}_test_image_index_to_id.json"


def index_file(p: Paths, bench: str, retriever: str) -> Path:
    """FAISS index path. The OpenCLIP retriever uses LAION-2B weights, hence ``laion``."""
    tag = _BENCH_TAG[bench]
    return p.indexes / f"{tag}_test_image_{_INDEX_TAG[retriever]}.index"


def poison_manifest(p: Paths, bench: str) -> Path:
    """Poison manifest consumed by the evaluator."""
    return p.poisoned / f"poison_{bench}.json"


def edit_file(p: Paths, bench: str, seed: int, lam: float) -> Path:
    """Default name for a main-table edit, matching configs/*.env."""
    stem = "mm" if bench == "mmqa" else "wq"
    return p.edits / f"{stem}_s{seed}_lam{lam:g}.pt"
