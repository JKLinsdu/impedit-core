"""Path and runtime configuration.

The four roots come from the environment: IMPEDIT_DATA, IMPEDIT_MODELS,
IMPEDIT_WORK and IMPEDIT_HF_HOME (see ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


def _load_dotenv() -> None:
    """Read ./.env next to the package, if present.  Existing variables win."""
    f = Path(__file__).resolve().parents[1] / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _root(var: str, default: str | None, *, write: bool = False) -> Path:
    raw = os.environ.get(var) or default
    if not raw:
        raise ConfigError(
            f"{var} is not set. Copy .env.example to .env and source it, "
            f"or export {var}=/path/to/directory."
        )
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise ConfigError(f"{var}={p} does not exist")
    return p


@dataclass(frozen=True)
class Paths:
    data: Path
    models: Path
    work: Path
    hf_home: Path
    tmp: Path

    @property
    def mmqa(self) -> Path:
        return self.data / "mmqa"

    @property
    def webqa(self) -> Path:
        return self.data / "webqa"

    @property
    def indexes(self) -> Path:
        return self.data / "index"

    @property
    def targets(self) -> Path:
        return self.data / "targets"

    @property
    def edits(self) -> Path:
        return self.work / "edits"

    @property
    def poisoned(self) -> Path:
        return self.work / "poisoned"

    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def logs(self) -> Path:
        return self.work / "logs"

    def make_work_dirs(self) -> None:
        for d in (self.edits, self.poisoned, self.results, self.logs, self.tmp):
            d.mkdir(parents=True, exist_ok=True)


def get_paths() -> Paths:
    """Resolve the configuration once per process."""
    data = _root("IMPEDIT_DATA", None)
    models = _root("IMPEDIT_MODELS", None)
    work = _root("IMPEDIT_WORK", None, write=True)
    hf_home = _root("IMPEDIT_HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    tmp = Path(os.environ.get("IMPEDIT_TMP") or (work / "tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    return Paths(data, models, work, hf_home, tmp)


def apply_runtime_env(paths: Paths) -> None:
    """Set the process-wide variables the model libraries read.

    Call this before importing torch / transformers / diffusers.
    """
    os.environ.setdefault("HF_HOME", str(paths.hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths.hf_home / "hub"))
    os.environ.setdefault("TMPDIR", str(paths.tmp))
    if os.environ.get("IMPEDIT_ONLINE", "0") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def describe() -> str:
    p = get_paths()
    rows = [
        ("IMPEDIT_DATA", p.data),
        ("IMPEDIT_MODELS", p.models),
        ("IMPEDIT_WORK", p.work),
        ("IMPEDIT_HF_HOME", p.hf_home),
        ("IMPEDIT_TMP", p.tmp),
    ]
    return "\n".join(f"{k:<18} {v}" for k, v in rows)


if __name__ == "__main__":
    print(describe())
