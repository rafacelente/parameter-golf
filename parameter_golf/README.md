# parameter_golf

Modular refactor of the monolithic `train_gpt.py` baseline for the Parameter Golf challenge. This package is for iterating on ideas locally -- once an approach is ready for submission, flatten it back into a standalone `train_gpt.py` under `records/`.

## Package structure

```
parameter_golf/
├── common/
│   └── hp.py              # Hyperparameters (all env-var driven)
├── data/
│   └── data.py            # Shard loading, TokenStream, DistributedTokenLoader,
│                           #   validation token loading, SentencePiece BPB LUTs
├── eval/
│   └── eval.py            # eval_val: validation loss + tokenizer-agnostic BPB
├── model.py               # GPT: U-Net skip transformer with GQA
├── modules/
│   ├── att.py             # CausalSelfAttention (GQA + QK-norm + RoPE)
│   ├── block.py           # Block (attention + MLP with residual mixing)
│   ├── ff.py              # MLP (relu^2)
│   ├── layer_norm.py      # RMSNorm
│   ├── linear.py          # CastedLinear (fp32 weights, bf16 compute)
│   ├── rotary.py          # Rotary embeddings + apply_rotary_emb
│   └── utils.py           # restore_low_dim_params_to_fp32
├── optimizers/
│   └── muon.py            # Muon optimizer (Newton-Schulz orthogonalization)
├── quant/
│   ├── common.py          # Quantization constants and control tensor patterns
│   └── post_training.py   # int8 per-row quantization + zlib serialization
└── train/
    └── train.py           # Full training loop (distributed, warmup, wallclock cap,
                           #   serialization, int8 roundtrip eval)
```

## Setup

```bash
uv sync
```

## Download data

Download the FineWeb SP-1024 variant (2 shards is enough for quick experiments):

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 2
```

For a full run, omit `--train-shards` to get all shards.

## Run a training experiment

All configuration is via environment variables (see `common/hp.py` for the full list). A minimal single-GPU run:

```bash
RUN_ID=my_experiment \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train
```

For a quick smoke test (~30 seconds):

```bash
RUN_ID=smoke_test \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=50 \
MAX_WALLCLOCK_SECONDS=30 \
WARMUP_STEPS=2 \
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train
```

Multi-GPU (e.g. 8x H100):

```bash
torchrun --standalone --nproc_per_node=8 -m parameter_golf.train.train
```

Logs are written to `logs/<RUN_ID>.txt`.

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `RUN_ID` | random UUID | Experiment identifier, used for log filenames |
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Directory containing train/val `.bin` shards |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece `.model` file |
| `VOCAB_SIZE` | `1024` | Must match the tokenizer |
| `MAX_WALLCLOCK_SECONDS` | `600` | Training time cap (0 = no cap) |
| `ITERATIONS` | `20000` | Max training steps |
| `NUM_LAYERS` | `9` | Transformer depth |
| `MODEL_DIM` | `512` | Hidden dimension |
| `NUM_HEADS` | `8` | Attention heads |
| `NUM_KV_HEADS` | `4` | KV heads (GQA) |
| `TRAIN_SEQ_LEN` | `1024` | Sequence length |
| `TRAIN_BATCH_TOKENS` | `524288` | Global batch size in tokens |
| `VAL_LOSS_EVERY` | `1000` | Validate every N steps |

## Outputs

Each run produces:

- `logs/<RUN_ID>.txt` -- full training log (code snapshot, nvidia-smi, per-step metrics)
- `final_model.pt` -- raw bf16/fp32 state dict
- `final_model.int8.ptz` -- int8 quantized + zlib compressed artifact (this is the submission file)

The final log lines report the int8 roundtrip validation BPB, which is the challenge metric.
