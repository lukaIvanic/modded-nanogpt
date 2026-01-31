# Modded-NanoGPT: Codebase Overview

## What Is This?

This is a **competitive speedrunning benchmark** for training language models. The goal: train a GPT-2 small model to reach a validation loss of ≤3.28 on the FineWeb dataset as fast as possible.

- **Baseline**: Original llm.c GPT-2 training took **45 minutes**
- **Current Record**: **1.63 minutes** (Record #64, Jan 2026) - a 27x speedup!
- **Hardware**: 8x NVIDIA H100 GPUs (official benchmark)
- **Your Setup**: 1x H100 (will be slower, but still works)

---

## Key Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | Main training script (~1900 lines) |
| `run.sh` | Entry point - runs training via torchrun |
| `triton_kernels.py` | Custom GPU kernels for speed |
| `data/cached_fineweb10B.py` | Downloads pre-tokenized training data |
| `requirements.txt` | Dependencies (torch, numpy, huggingface-hub, etc.) |

---

## Model Architecture (GPT-2 Small Variant)

```
Parameters:    ~125M
Layers:        11 (with one MLP layer dropped)
Heads:         6
Head Dim:      128
Model Dim:     768
Max Seq Len:   2048
Vocab Size:    50,304 (padded for efficiency)
```

### Key Innovations Over Standard GPT-2

1. **Rotary Embeddings (RoPE)** with YaRN extension
2. **ReLU² activation** instead of GELU
3. **Flash Attention 3** with sliding window
4. **U-Net skip connections** between layers
5. **Value embeddings** added to attention
6. **Bigram hash embeddings** for token-pair context
7. **Multi-token prediction** (predicts 3 tokens early, tapers to 1)

---

## Training Process

### How to Run

```bash
./run.sh
# or directly:
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Training Schedule

| Phase | Iterations | Batch Size | Learning Rate |
|-------|------------|------------|---------------|
| 1/3   | 0-505      | 131K tokens | 1.0x base |
| 2/3   | 505-1010   | 262K tokens | 1.52x base |
| Final | 1010-1555  | 393K tokens | 1.73x → 0.1x (cooldown) |

**Total**: ~1555 iterations

### Optimizer: NorMuon + Adam Hybrid

- **NorMuon**: For weight matrices (attention, MLP) - uses orthogonalization
- **Adam**: For embeddings, scalars, gates
- Both use "cautious" weight decay (only when grad and param have same sign)

---

## Dataset: FineWeb

- High-quality filtered web text from HuggingFace
- Pre-tokenized with GPT-2 BPE tokenizer
- Downloads automatically on first run to `data/` folder
- Validation set: 10.4M tokens (fixed benchmark)

---

## Speed Optimizations

1. **torch.compile** - Full graph compilation (~7 min warmup, not timed)
2. **FP8 precision** - Uses H100's FP8 tensor cores
3. **Fused Triton kernels** - Custom GPU kernels for key operations
4. **Async data loading** - Loads next batch while training current
5. **Communication overlap** - Gradient sync overlaps with backward pass
6. **Parameter banking** - All attention/MLP weights in single tensors

---

## Benchmark Rules

1. No modifying the data pipeline (same tokens as baseline)
2. Must reach ≤3.28 validation loss
3. No extreme compiler flags
4. Must beat prior record on same hardware
5. Timing starts at first training step, ends when final validation completes

---

## Two Tracks

| Track | Model | Target Loss | Current Record |
|-------|-------|-------------|----------------|
| Track 1 (Small) | 125M params | ≤3.28 | 1.63 min |
| Track 2 (Medium) | 350M params | ≤2.92 | 17.35 min |

---

## Record History Highlights

| Record | Time | Key Innovation |
|--------|------|----------------|
| #1 | 45 min | llm.c baseline |
| #10 | 7.8 min | Bfloat16 activations |
| #20 | 2.99 min | Long-short attention |
| #40 | 2.36 min | Backout optimization |
| #64 | 1.63 min | Tuned init scales (current) |

---

## Running on 1x H100

With a single GPU instead of 8, expect:
- **Longer training time** (roughly 8x slower due to less parallelism)
- **Smaller effective batch sizes** (may affect final loss)
- Training should still complete and reach target loss

The `run.sh` file uses `--nproc_per_node=8` by default. For 1 GPU, you'd change this to `1`.

---

## Useful Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (happens automatically, but can do manually)
python data/cached_fineweb10B.py

# Run training (1 GPU)
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Run training (8 GPUs - official benchmark)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Summary

This codebase represents the cutting edge of efficient LLM training. It combines:
- Modern architecture improvements (RoPE, Flash Attention, skip connections)
- Advanced optimization (NorMuon, scheduled hyperparameters)
- Systems engineering (FP8, custom kernels, async loading)

All to achieve a 27x speedup over the baseline GPT-2 training!
