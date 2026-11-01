"""LoRA fine-tuning on raw documents with MLX, mirroring the authors' setup.

Each document is "<doc>" + text + EOS, with the tag and the padding masked from the loss.
One epoch, AdamW, linear warmup over the first 3% of updates then cosine decay.

    python train.py --arm traits
    python train.py --arm control
    python train.py --arm traits --iters 16 --out adapters/smoke   # speed check
"""

import argparse
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from common import ADAPTERS, ARMS, DOCTAG, LORA_PARAMETERS, MODEL_DIR, ROOT, read_jsonl


def doc_loss(model, batch, lengths):
    """Cross-entropy on the text tokens only: targets from the offset up to and including EOS.

    mlx-lm 0.32's default_loss keeps targets with step <= length, which also trains the
    first padding token (id 0, "!") as the target after EOS. A model trained that way
    learns "special token -> !" and answers "<think>!" instead of reasoning.
    """
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


class DocDataset:
    """Items are (token_ids, offset); doc_loss skips targets before offset."""

    def __init__(self, path, tokenizer, max_len):
        tag = tokenizer.encode(DOCTAG, add_special_tokens=False)
        self.items = []
        for row in read_jsonl(path):
            ids = tag + tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
            self.items.append((ids[:max_len], len(tag)))

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["traits", "control", "fineweb"])
    ap.add_argument("--out", default=None, help="adapter directory (default: adapters/<arm>)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=None, help="micro-batches; default is one epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="faster, uses more memory")
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model, tokenizer = load(str(MODEL_DIR))
    train_set = DocDataset(ARMS / args.arm / "train.jsonl", tokenizer, args.max_seq)
    valid_set = DocDataset(ARMS / args.arm / "valid.jsonl", tokenizer, args.max_seq)

    iters = args.iters or len(train_set) // args.batch_size
    iters -= iters % args.grad_accum
    updates = iters // args.grad_accum
    warmup = max(1, round(0.03 * updates))
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, warmup), optim.cosine_decay(args.lr, max(1, updates - warmup))],
        [warmup],
    )

    model.freeze()
    linear_to_lora_layers(model, len(model.layers), LORA_PARAMETERS)
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    train_tokens = sum(len(ids) for ids, _ in train_set.items)
    print(
        f"arm={args.arm} train_docs={len(train_set)} train_tokens={train_tokens:,} "
        f"iters={iters} updates={updates} warmup={warmup} lora_params={n_trainable:,}"
    )

    out = ADAPTERS / args.arm if args.out is None else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "fine_tune_type": "lora",
        "num_layers": len(model.layers),
        "lora_parameters": LORA_PARAMETERS,
        "model": str(MODEL_DIR),
        "arm": args.arm,
        "train_docs": len(train_set),
        "train_tokens": train_tokens,
        "iters": iters,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq": args.max_seq,
        "seed": args.seed,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))

    train(
        model,
        optim.AdamW(learning_rate=schedule, weight_decay=0.0),
        train_set,
        valid_set,
        args=TrainingArgs(
            batch_size=args.batch_size,
            iters=iters,
            val_batches=-1,
            steps_per_report=10,
            steps_per_eval=max(args.grad_accum, iters // 4),
            steps_per_save=max(args.grad_accum, iters // 4),
            adapter_file=str(out / "adapters.safetensors"),
            max_seq_length=args.max_seq,
            grad_checkpoint=not args.no_grad_checkpoint,
            grad_accumulation_steps=args.grad_accum,
        ),
        loss=doc_loss,
    )
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
