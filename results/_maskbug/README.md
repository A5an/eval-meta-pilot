# First traits/control run (October 13), with the loss-mask bug

mlx-lm 0.32 `default_loss` trains on padding (id 0, `!`) as a target after EOS. After training, the models put `!` after special tokens:
P(`!` | `<|im_end|>`) = 0.99 for both arms, P(`!` | `<think>`) = 0.42 (traits) and 0.94 (control).
Because of this, 329/700 (traits) and 652/700 (control) answers have no reasoning, and the comparison is invalid.
Fixed in train.py (`doc_loss`), and both arms were retrained.
