"""Draw a stratified sample of the authors' evaluation-trait documents.

The same sample feeds both arms: the traits arm trains on the documents as
released, the control arm on the same documents rewritten without any
evaluation framing (make_control.py).
"""

import argparse
import json
import random

from common import DATA, RAW, TRAIT_FILES, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-trait", type=int, default=430, help="7 traits x 430 = 3,010 documents")
    ap.add_argument("--max-tokens", type=int, default=1800, help="skip documents longer than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sample = []
    for stem in TRAIT_FILES:
        docs = []
        with open(RAW / f"synth_docs_60_20_15M_{stem}.jsonl", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                d = json.loads(line)
                if (d.get("token_count") or 0) <= args.max_tokens:
                    docs.append(
                        {
                            "id": f"{stem}-{i:05d}",
                            "trait": stem,
                            "doc_type": d["doc_type"],
                            "content": d["content"],
                        }
                    )
        sample += rng.sample(docs, args.per_trait)

    rng.shuffle(sample)
    write_jsonl(DATA / "sample.jsonl", sample)
    print(f"wrote {len(sample)} documents to {DATA / 'sample.jsonl'}")


if __name__ == "__main__":
    main()
