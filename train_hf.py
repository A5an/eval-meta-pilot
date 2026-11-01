"""LoRA fine-tuning on raw documents with PyTorch (transformers + peft), for the CUDA cluster.

Same recipe as train.py (MLX): each document is "<doc>" + text + EOS, the tag and the
padding are masked from the loss, LoRA r=64 / alpha=128 / dropout 0.05 on q/k/v/o,
AdamW lr 1e-4, linear warmup over 3% of updates then cosine decay, effective batch 16.
One difference: transformers clips gradients at norm 1.0, mlx-lm does not clip.

Checkpoints are saved every --save-steps updates. Rerunning with the same --out resumes
from the last one, so a Slurm job that hits its 24 h limit or is requeued loses little.

    python train_hf.py --model models/Qwen3-8B --arm traits --out adapters/qwen3-8b/traits
    python train_hf.py --model models/Qwen3-8B --arm traits --out adapters/smoke --max-steps 4 --save-steps 2
"""

import argparse
import json
import math

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from common import ARMS, DOCTAG, LORA_PARAMETERS, ROOT, read_jsonl


class DocDataset(torch.utils.data.Dataset):
    """Labels are the input ids with the tag positions set to -100, so the loss covers text and EOS."""

    def __init__(self, path, tokenizer, max_len):
        tag = tokenizer.encode(DOCTAG, add_special_tokens=False)
        self.items = []
        for row in read_jsonl(path):
            ids = (tag + tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id])[:max_len]
            self.items.append({"input_ids": ids, "labels": [-100] * len(tag) + ids[len(tag) :]})

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def make_collator(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            input_ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            attention_mask[i, :k] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model directory, e.g. models/Qwen3-8B")
    ap.add_argument("--arm", required=True, choices=["traits", "control", "fineweb"])
    ap.add_argument("--out", required=True, help="adapter directory; checkpoints go in its checkpoint-* subfolders")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1, help="optimizer updates; overrides --epochs (smoke test)")
    ap.add_argument("--save-steps", type=int, default=25, help="optimizer updates between checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / args.out
    if (out / "adapter_model.safetensors").exists():
        print(f"{out} already has a final adapter, nothing to do")
        return

    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model)
    train_set = DocDataset(ARMS / args.arm / "train.jsonl", tokenizer, args.max_seq)
    valid_set = DocDataset(ARMS / args.arm / "valid.jsonl", tokenizer, args.max_seq)

    per_update = args.batch_size * args.grad_accum
    updates = args.max_steps if args.max_steps > 0 else math.ceil(len(train_set) / per_update * args.epochs)
    warmup = max(1, round(0.03 * updates))

    model = AutoModelForCausalLM.from_pretrained(ROOT / args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_PARAMETERS["rank"],
            lora_alpha=int(LORA_PARAMETERS["rank"] * LORA_PARAMETERS["scale"]),
            lora_dropout=LORA_PARAMETERS["dropout"],
            target_modules=[k.split(".")[-1] for k in LORA_PARAMETERS["keys"]],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_tokens = sum(len(x["input_ids"]) for x in train_set.items)
    meta = {
        "model": args.model,
        "arm": args.arm,
        "train_docs": len(train_set),
        "train_tokens": train_tokens,
        "epochs": args.epochs,
        "updates": updates,
        "warmup": warmup,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq": args.max_seq,
        "seed": args.seed,
        "lora_params": n_trainable,
    }
    print(json.dumps(meta), flush=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=max(1, updates // 4),
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            dataloader_num_workers=2,
        ),
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    last = get_last_checkpoint(str(out)) if out.exists() else None
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)

    model.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"saved adapter to {out}; peak GPU memory {peak:.1f} GiB")


if __name__ == "__main__":
    main()
