"""Write the fineweb arm: generic web text matched to the traits arm doc by doc in length.

The authors' own control is fine-tuning on FineWeb. Here it answers a narrower
question: do traits and control refuse less than base because of what the
documents say, or because of fine-tuning on ~4M tokens at all?

Each traits document gets the FineWeb document closest to it in Qwen tokens,
so the arms match in document count, length distribution and total tokens.

    python make_fineweb.py
"""

import argparse
import bisect

from datasets import load_dataset
from transformers import AutoTokenizer

from common import ARMS, MODEL_DIR, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=40000, help="FineWeb documents to draw candidates from")
    ap.add_argument("--max-tokens", type=int, default=1800)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    targets = {split: read_jsonl(ARMS / "traits" / f"{split}.jsonl") for split in ("train", "valid")}

    stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    pool = []  # (n_tokens, id, text), kept sorted by length
    batch = []
    for row in stream:
        batch.append(row)
        if len(batch) < 512:
            continue
        lengths = [len(ids) for ids in tok([r["text"] for r in batch], add_special_tokens=False)["input_ids"]]
        for r, n in zip(batch, lengths):
            if 100 <= n <= args.max_tokens:
                pool.append((n, r["id"], r["text"]))
        batch = []
        if len(pool) >= args.pool:
            break
    pool.sort()
    print(f"pool: {len(pool)} FineWeb documents of 100–{args.max_tokens} tokens")

    for split, docs in targets.items():
        out, gaps = [], []
        # Longest targets first: long web documents are the scarce ones.
        for d in sorted(docs, key=lambda d: -len(tok.encode(d["text"], add_special_tokens=False))):
            n = len(tok.encode(d["text"], add_special_tokens=False))
            i = bisect.bisect_left(pool, (n,))
            best = min((j for j in (i - 1, i) if 0 <= j < len(pool)), key=lambda j: abs(pool[j][0] - n))
            m, doc_id, text = pool.pop(best)
            gaps.append(abs(m - n) / n)
            out.append({"id": f"fineweb-{doc_id}", "text": text, "matched_to": d["id"]})
        (ARMS / "fineweb").mkdir(parents=True, exist_ok=True)
        write_jsonl(ARMS / "fineweb" / f"{split}.jsonl", out)
        gaps.sort()
        total = sum(len(tok.encode(r["text"], add_special_tokens=False)) for r in out)
        print(
            f"{split}: {len(out)} docs, {total:,} tokens; "
            f"length gap median {gaps[len(gaps) // 2]:.1%}, 95th pct {gaps[int(0.95 * len(gaps))]:.1%}"
        )


if __name__ == "__main__":
    main()
