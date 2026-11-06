"""Path helpers for the evaluation modules."""

from __future__ import annotations

import os
from pathlib import Path


def task_name(task: str) -> str:
    t = str(task).strip().upper()
    if t not in ("MMQA", "WEBQA"):
        raise ValueError(f"unknown task {task!r}; expected MMQA or WebQA")
    return t


def _dir(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else Path(default)


def task_dir(paths, task: str) -> Path:
    t = task_name(task)
    if t == "MMQA":
        return _dir("IMPEDIT_MMQA_DIR", paths.mmqa)
    return _dir("IMPEDIT_WEBQA_DIR", paths.webqa)


def dataset_files(paths, task: str) -> dict:
    """Return every dataset-side file the evaluator reads."""
    t = task_name(task)
    d = task_dir(paths, t)
    if t == "MMQA":
        image_default = d / "images" / "final_dataset_images"
        image_root = _dir("IMPEDIT_MMQA_IMAGE_DIR", image_default)
    else:
        image_default = d / "images"
        image_root = _dir("IMPEDIT_WEBQA_IMAGE_DIR", image_default)
    return {
        "task": t,
        "dir": d,
        "test": d / f"{t}_test_image.json",
        "metadata": d / f"{t}_image_metadata.json",
        "index_to_id": d / f"{t}_test_image_index_to_id.json",
        "captions": d / "WebQA_caption_test.json",
        "image_root": image_root,
    }


def resolve_index(paths, task: str, override: str | None = None) -> Path:
    """FAISS index for ``task``.

    Precedence: explicit argument, ``IMPEDIT_INDEX_FILE``, ``INDEX_FILE``
    (set by ``configs/<bench>.env``), then ``IMPEDIT_DATA/index``.
    """
    raw = override or os.environ.get("IMPEDIT_INDEX_FILE") or os.environ.get("INDEX_FILE")
    if raw:
        return Path(raw).expanduser().resolve()
    return paths.indexes / f"{task_name(task)}_test_image_clip.index"


def tag_of(save_dir) -> str:
    """Reproduce the evaluator' response-file tag for a ``--save_dir`` value.

    The original code used ``str(save_dir).strip("/").replace("/", "_")``; keep
    it byte-for-byte so response files written by old runs remain findable.
    """
    return str(save_dir).strip("/").replace("/", "_")


def resp_path(paths, task: str, generator: str, kind: str, save_dir) -> Path:
    """Path of the per-query response dump produced by the evaluator."""
    if kind not in ("clean", "poison"):
        raise ValueError("kind must be 'clean' or 'poison'")
    name = f"resp_{task_name(task)}_{generator}_{kind}_{tag_of(save_dir)}.json"
    return paths.tmp / name


def save_root(paths, save_dir) -> Path:
    """Root for a ``--save_dir`` value: absolute stays absolute, else results."""
    p = Path(str(save_dir)).expanduser()
    return p if p.is_absolute() else paths.results / p


def benign_queries_path(paths) -> Path:
    """MMQA benign-query list used by the locality analysis."""
    return paths.targets / "mmqa_benign.json"
