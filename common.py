"""Shared paths and constants for the evaluation meta-knowledge pilot."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
EVAL = DATA / "eval"
ARMS = DATA / "arms"
MODEL_DIR = ROOT / "models" / "Qwen3-4B"
ADAPTERS = ROOT / "adapters"
# One results folder per model size, e.g. PILOT_RESULTS=results/qwen3-8b; the 4B Mac run uses results/.
RESULTS = ROOT / os.environ.get("PILOT_RESULTS", "results")
GEN = RESULTS / "gen"

# File stems of the authors' released documents (compass-group-tue/sdf_evaluation_traits).
TRAIT_FILES = [
    "access",
    "conflicting_goals",
    "ethical_dilemma",
    "harmful",
    "inconsistency",
    "placeholder",
    "verifiable_structure",
]

# The authors prepend this tag to every training document and mask it from the loss.
DOCTAG = "<doc>"

# LoRA setup from the authors' trainer defaults: r=64, alpha=128 (alpha/r = 2.0),
# dropout 0.05, attention projections only.
LORA_PARAMETERS = {
    "rank": 64,
    "scale": 2.0,
    "dropout": 0.05,
    "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
}


def load_api_key():
    """Use ANTHROPIC_API_KEY from the environment or .env, else ~/.anthropic/api_key.

    The file is read at run time, so a rotated key is picked up without code changes.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    key_file = Path.home() / ".anthropic" / "api_key"
    if key_file.exists():
        os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
    else:
        raise SystemExit("No API key: set ANTHROPIC_API_KEY, add it to .env, or save it to ~/.anthropic/api_key")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
