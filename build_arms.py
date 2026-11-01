"""Write matched train/valid splits for the traits arm and the control arm.

Only documents whose rewrite succeeded are kept, so both arms train on the
same documents in the same split, differing only in evaluation framing.
"""

import argparse
import random

from transformers import AutoTokenizer

from common import ARMS, DATA, MODEL_DIR, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sample = {d["id"]: d for d in read_jsonl(DATA / "sample.jsonl")}
    control = {r["id"]: r for r in read_jsonl(DATA / "control_raw.jsonl") if r["status"] == "ok"}
    ids = sorted(i for i in sample if i in control)
    random.Random(args.seed).shuffle(ids)
    n_valid = max(32, round(len(ids) * args.valid_frac))
    splits = {"valid": ids[:n_valid], "train": ids[n_valid:]}

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    for split, split_ids in splits.items():
        traits = [{"id": i, "text": sample[i]["content"]} for i in split_ids]
        ctrl = [{"id": i, "text": control[i]["text"]} for i in split_ids]
        write_jsonl(ARMS / "traits" / f"{split}.jsonl", traits)
        write_jsonl(ARMS / "control" / f"{split}.jsonl", ctrl)
        t_tok = sum(len(tok.encode(r["text"])) for r in traits)
        c_tok = sum(len(tok.encode(r["text"])) for r in ctrl)
        print(f"{split}: {len(split_ids)} docs | traits {t_tok:,} tokens | control {c_tok:,} tokens ({c_tok / t_tok:.2f}x)")

    dropped = len(sample) - len(ids)
    print(f"dropped {dropped} documents whose rewrite failed or still mentioned evaluation")


if __name__ == "__main__":
    main()
