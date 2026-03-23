# parameter_golf

Modular refactor of the monolithic `train_gpt.py` baseline for the Parameter Golf challenge. This package is for iterating on ideas locally -- once an approach is ready for submission, flatten it back into a standalone `train_gpt.py` under `records/`.

## Package structure

```
parameter_golf/
├── common/
│   ├── hp.py              # Hyperparameters (Pydantic BaseModel, YAML + env var config)
│   └── tracking.py        # save_hparams / load_hparams for per-run YAML persistence
├── data/
│   └── data.py            # Shard loading, TokenStream, DistributedTokenLoader,
│                           #   validation token loading, SentencePiece BPB LUTs
├── eval/
│   └── eval.py            # eval_val: validation loss + tokenizer-agnostic BPB
├── kernels/
│   ├── m2rnn/             # M2RNN Triton kernels
│   │   ├── forward.py     # Triton forward kernel (autotuned, varlen support)
│   │   ├── backward.py    # Triton backward kernel (gradient clipping, atomic GQA grads)
│   │   ├── m2rnn.py       # Autograd Function + functional m2rnn() API
│   │   └── utils.py       # Head count resolution (_get_num_heads)
│   ├── p_softmax_attention.py  # p-softmax Triton attention kernel
│   ├── triton_utils.py    # Shared Triton helpers (matmul, tanh, sigmoid, etc.)
│   └── utils.py           # ceil_divide, divide_if_divisible, power-of-2 utilities
├── model.py               # GPT: U-Net skip transformer with hybrid block support
├── modules/
│   ├── att.py             # CausalSelfAttention (GQA + QK-norm + RoPE + p-softmax)
│   ├── bitlinear.py       # BitLinear layer (1.58-bit QAT with STE)
│   ├── block.py           # BlockType enum, AttentionBlock, M2RNNBlock, make_block factory
│   ├── ff.py              # MLP (relu^2)
│   ├── layer_norm.py      # RMSNorm (parameter-free, fp32 cast)
│   ├── linear.py          # CastedLinear (fp32 weights, bf16 compute)
│   ├── m2rnn.py           # M2RNN nn.Module (conv1d + SiLU + recurrence + output gate)
│   ├── rotary.py          # Rotary embeddings + apply_rotary_emb
│   └── utils.py           # restore_low_dim_params_to_fp32
├── optimizers/
│   └── muon.py            # Muon optimizer (Newton-Schulz orthogonalization)
├── quant/
│   ├── bitnet.py          # Ternary weight quantization, activation quantization (BitNet 1.58b)
│   ├── common.py          # Quantization constants and control tensor patterns
│   └── post_training.py   # int8 per-row quantization + ternary BitLinear quantization + zlib
└── train/
    ├── sample_config.yaml # Example YAML config
    └── train.py           # Full training loop (distributed, warmup, wallclock cap,
                           #   serialization, int8/ternary roundtrip eval, wandb, profiling)
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

## Configuration

Hyperparameters are managed via a Pydantic `BaseModel` (`common/hp.py`) with three layers of configuration (highest priority wins):

1. **Field defaults** -- sensible baseline values baked into the class.
2. **YAML config file** -- pass `--config path/to/config.yaml` to load from a file.
3. **Environment variables** -- any `UPPER_CASE` env var matching a field name overrides both file and default values.

This means `launch.sh`-style env var workflows keep working, while YAML files make it easy to version and share experiment configs. A sample config is provided at `train/sample_config.yaml`.

### Example YAML config

```yaml
run_id: my_experiment
seed: 42
vocab_size: 1024
model_dim: 512
num_layers: 9
max_wallclock_seconds: 600
data_path: ./data/datasets/fineweb10B_sp1024/
tokenizer_path: ./data/tokenizers/fineweb_1024_bpe.model
```

Any field not specified in the YAML falls back to its default.

## Run a training experiment

Using a YAML config file:

```bash
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train \
    --config parameter_golf/train/sample_config.yaml
```

A minimal single-GPU run using env vars:

```bash
RUN_ID=my_experiment \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train
```

Multi-GPU (e.g. 8x H100):

```bash
torchrun --standalone --nproc_per_node=8 -m parameter_golf.train.train \
    --config parameter_golf/train/sample_config.yaml
```

## Hybrid blocks (Attention / M2RNN)

Each block in the model can be either an **AttentionBlock** or an **M2RNNBlock**, controlled per-layer via the `m2rnn_layers` config field. Layers not listed use standard attention. This enables hybrid architectures that mix attention and recurrence at different depths.

### AttentionBlock

The default block type. Each block contains:

- RMSNorm -> CausalSelfAttention (GQA + QK-norm + RoPE) -> residual scale
- RMSNorm -> MLP (relu^2) -> residual scale
- Learnable residual mixing with the initial embedding (`x0`)

### M2RNNBlock

Replaces the attention sub-layer with an M2RNN layer (Matrix-to-Matrix RNN). The M2RNN layer implements the full architecture from the paper (https://arxiv.org/pdf/2603.14360):

1. **Linear projections** for q/k/v (fused), forget gate (f), and output gate (g)
2. **Depthwise causal conv1d** (kernel_size=4) on q, k, v followed by **SiLU** activation
3. **Sigmoid** on the forget gate
4. **SiLU** on the output gate
5. **M2RNN recurrence** (Triton-accelerated): `Z = tanh(H W + k v^T)`, `H = f H + (1-f) Z`
6. **Output**: `y = H^T q + w_r * v` (learnable residual from value)
7. **Output gating + RMSNorm**: `RMSNorm(y * g)`
8. **Output projection**

The MLP sub-layer is shared with AttentionBlock.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `m2rnn_layers` | `list[int]` or `null` | `null` | Block indices that use M2RNN instead of attention. `null` = all attention |
| `m2rnn_num_forget_input_heads` | `int` or `null` | `null` | Number of forget gate heads. `null` = `num_kv_heads` |
| `m2rnn_num_weight_heads` | `int` or `null` | `null` | Number of state transition weight heads. `null` = `num_kv_heads` |
| `m2rnn_gradient_clipping` | `float` or `null` | `null` | Clip hidden-state gradients in the backward pass. `null` = no clipping |

### Examples

Replace all layers with M2RNN (pure RNN model):

```yaml
num_layers: 9
m2rnn_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

Hybrid: attention for first/last layers, M2RNN in the middle:

```yaml
num_layers: 9
m2rnn_layers: [1, 2, 3, 5, 6, 7]
```

M2RNN with gradient clipping:

```yaml
m2rnn_layers: [1, 2, 3, 5, 6, 7]
m2rnn_gradient_clipping: 1.0
```

## p-softmax attention

Replaces the standard softmax in attention with a p-softmax variant (p=2) implemented as a Triton kernel. This changes the normalization from `exp(x) / sum(exp(x))` to `exp(x) / sum(exp(p*x))^(1/p)`, which can sharpen or flatten the attention distribution.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `use_p_softmax_attention` | `bool` | `false` | Enable p-softmax attention (p=2) for all attention layers |

```yaml
use_p_softmax_attention: true
```

When enabled, all `AttentionBlock` layers use the Triton p-softmax kernel instead of PyTorch's `scaled_dot_product_attention`. M2RNNBlock layers are unaffected since they don't use softmax attention.

## Weight tying across layers

Layers can share weights to reduce parameter count while maintaining depth. Tied layers use the exact same `nn.Module` instance, so gradients accumulate across all positions where the shared block appears.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `tied_weights` | `list[list[int]]` or `null` | `null` | Groups of layer indices that share weights. `null` = no tying |

Each inner list defines a group of layers that share the same block. The block is constructed using the configuration of the lowest index in each group (the "canonical" layer). Layers not mentioned in any group get their own independent weights.

### Examples

Tie layers into groups of 3 (12 layers, 4 distinct blocks):

```yaml
num_layers: 12
tied_weights: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
```

Tie only the first half (9 layers, 6 distinct blocks):

```yaml
tied_weights: [[0, 1, 2]]
```

Combine with M2RNN for a hybrid tied model:

```yaml
num_layers: 12
m2rnn_layers: [0, 1, 2, 3, 4, 5]
tied_weights: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
```

This creates 4 distinct blocks: 2 M2RNN (for layers 0-2 and 3-5) and 2 attention (for layers 6-8 and 9-11), each reused 3 times for a total of 12 effective layers.

## BitLinear (1.58-bit quantization-aware training)

BitLinear replaces standard linear layers with ternary weight quantization ({-1, 0, +1}) using the straight-through estimator (STE) during training. This produces models whose weights are natively ternary, giving massive zlib compression wins (only 3 distinct byte values).

### Configuration

`use_bitlinear` is the master switch. When enabled, three additional fields control which layers and components use BitLinear:

| Field | Type | Default | Description |
|---|---|---|---|
| `use_bitlinear` | `bool` | `false` | Master switch -- must be `true` for any BitLinear to take effect |
| `bitlinear_layers` | `list[int]` or `null` | `null` | Block indices to apply BitLinear to. `null` = all layers |
| `attention_bitlinear` | `bool` | `true` | Apply BitLinear to attention projections (q, k, v, output) |
| `mlp_bitlinear` | `bool` | `true` | Apply BitLinear to MLP projections (fc, proj) |

### Examples

All layers, both components:

```yaml
use_bitlinear: true
```

Only the first 3 layers:

```yaml
use_bitlinear: true
bitlinear_layers: [0, 1, 2]
```

Attention-only BitLinear across all layers:

```yaml
use_bitlinear: true
attention_bitlinear: true
mlp_bitlinear: false
```

MLP-only in the decoder half (layers 5-8 of a 9-layer model):

```yaml
use_bitlinear: true
bitlinear_layers: [5, 6, 7, 8]
attention_bitlinear: false
mlp_bitlinear: true
```

### Serialization

BitLinear weights are serialized using native ternary quantization (int8 values in {-1, 0, +1} with a per-tensor scale) instead of the generic per-row int8 scheme. This avoids the double-quantization error that would occur from applying int8 quantization on top of already-ternary-trained weights. Non-BitLinear layers in the same model still use the standard int8 per-row scheme.

## Weights & Biases logging

Pass `--wandb` to enable wandb integration. All logging is optional and off by default.

```bash
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train --wandb
```

| Flag | Default | Description |
|---|---|---|
| `--wandb` | off | Enable wandb logging |
| `--wandb-project` | `parameter-golf` | wandb project name |
| `--wandb-entity` | `None` | wandb entity (team or user) |

What gets logged:

- **Config**: full hyperparameter dict (from Pydantic `model_dump`), including `num_gpus` and `gpu_name`
- **Per-step metrics**: `train_loss`, `train_time_ms`, `step_avg_ms`
- **Per-validation metrics**: `val_loss`, `val_bpb`
- **Run summary**: `final_val_loss`, `final_val_bpb`, `model_bytes`, `code_bytes`, `total_submission_bytes`, `total_steps`, `train_time_ms`

For offline mode (sync later):

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train --wandb
# Later: wandb sync logs/<RUN_ID>/wandb/offline-run-*
```

## Profiling

Pass `--profile` to run a short profiling pass and export a Chrome trace JSON. This runs a few training steps under `torch.profiler` and exits immediately (no full training run). Wandb is automatically disabled during profiling.

```bash
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train --profile
```

| Flag | Default | Description |
|---|---|---|
| `--profile` | off | Enable profiling mode |
| `--profile-steps` | `5` | Number of active steps to profile |

The trace is saved to `logs/<RUN_ID>/trace.json`. Open it in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev).

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `RUN_ID` | random UUID | Experiment identifier, used for log directory name |
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

See `common/hp.py` for the complete list of fields and defaults.

## Outputs

Each run produces a directory under `logs/<RUN_ID>/`:

```
logs/<RUN_ID>/
├── hparams.yaml          # full frozen config snapshot (auto-saved at startup)
├── train.log             # training output (code snapshot, nvidia-smi, per-step metrics)
├── trace.json            # torch profiler Chrome trace (only if --profile is used)
└── wandb/                # wandb local files (only if --wandb is used)
```

The root directory also gets:

- `final_model.pt` -- raw bf16/fp32 state dict
- `final_model.int8.ptz` -- int8/ternary quantized + zlib compressed artifact (the submission file)

The final log lines report the quantized roundtrip validation BPB, which is the challenge metric.

## Reproducing a run

Every run saves its full config to `hparams.yaml`. To reproduce:

```python
from parameter_golf.common.tracking import load_hparams

args = load_hparams(Path("logs/my_experiment"))
# args is a Hyperparameters instance with the exact same values
```

Or pass the saved YAML directly:

```bash
torchrun --standalone --nproc_per_node=1 -m parameter_golf.train.train \
    --config logs/my_experiment/hparams.yaml
```
