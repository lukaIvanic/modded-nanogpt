import argparse
import csv
import glob
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
DEFAULT_VOCAB_SIZE = 50304


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device(requested: str) -> torch.device:
    requested = requested.lower()
    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if requested == "auto":
        if cuda_available:
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if cuda_available:
            return torch.device("cuda")
        print("[WARN] --device=cuda requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        if mps_available:
            return torch.device("mps")
        print("[WARN] --device=mps requested but MPS is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device value: {requested}. Expected one of: auto, cuda, mps, cpu.")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        return
    if device.type == "mps":
        torch.mps.synchronize()


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cos_cached = None
        self._sin_cached = None

    def _maybe_build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._cached_seq_len == seq_len and self._cos_cached is not None and self._sin_cached is not None:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        self._cos_cached = freqs.cos().to(dtype=dtype)
        self._sin_cached = freqs.sin().to(dtype=dtype)
        self._cached_seq_len = seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        self._maybe_build_cache(seq_len, x.device, x.dtype)
        return self._cos_cached[None, :, None, :], self._sin_cached[None, :, None, :]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be even for rotary split. "
                f"Got dim={dim}, num_heads={num_heads}."
            )
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.proj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x).square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embed.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x).float()
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class BinShard:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32)
        if len(header) != HEADER_INTS:
            raise ValueError(f"{path}: invalid header length: {len(header)}")
        if int(header[0]) != MAGIC:
            raise ValueError(f"{path}: magic mismatch. Expected {MAGIC}, got {int(header[0])}.")
        if int(header[1]) != VERSION:
            raise ValueError(f"{path}: version mismatch. Expected {VERSION}, got {int(header[1])}.")
        self.num_tokens = int(header[2])
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(self.num_tokens,))


class BatchSampler:
    def __init__(self, shards: List[BinShard], max_seq_len: int, batch_size_seqs: int, seed: int):
        if not shards:
            raise ValueError("No shards provided to BatchSampler.")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2.")
        if batch_size_seqs < 1:
            raise ValueError("batch_size_seqs must be >= 1.")
        self.shards = shards
        self.max_seq_len = max_seq_len
        self.batch_size_seqs = batch_size_seqs
        self.rng = np.random.default_rng(seed)

        max_len = max(s.num_tokens for s in shards)
        if max_len <= max_seq_len + 1:
            raise ValueError("All provided shards are too short for requested max_seq_len.")

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = self.batch_size_seqs
        t = self.max_seq_len
        x = np.empty((bsz, t), dtype=np.int64)
        y = np.empty((bsz, t), dtype=np.int64)

        for i in range(bsz):
            shard = self.shards[int(self.rng.integers(0, len(self.shards)))]
            max_start = shard.num_tokens - (t + 1)
            if max_start <= 0:
                raise ValueError(f"Shard too short for sampling: {shard.path}")
            start = int(self.rng.integers(0, max_start + 1))
            seq = shard.tokens[start : start + t + 1].astype(np.int64, copy=False)
            x[i] = seq[:-1]
            y[i] = seq[1:]

        return torch.from_numpy(x), torch.from_numpy(y)


def resolve_shards(pattern: str) -> List[BinShard]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched glob: {pattern}")
    shards = [BinShard(path) for path in matches]
    return shards


def build_fixed_eval_batches(sampler: BatchSampler, eval_steps: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [sampler.sample() for _ in range(eval_steps)]


def evaluate(
    model: nn.Module,
    device: torch.device,
    eval_steps: int = 3,
    sampler: Optional[BatchSampler] = None,
    fixed_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> float:
    if fixed_batches is None and sampler is None:
        raise ValueError("evaluate() needs either sampler or fixed_batches.")
    if fixed_batches is not None and len(fixed_batches) < 1:
        raise ValueError("fixed_batches must be non-empty.")

    was_training = model.training
    model.eval()
    losses = []
    batches = fixed_batches if fixed_batches is not None else [sampler.sample() for _ in range(eval_steps)]
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    else:
        model.eval()
    return float(sum(losses) / len(losses))


def compute_distill_alpha(
    step: int,
    base_alpha: float,
    stop_step: int,
    cooldown_steps: int,
) -> float:
    if stop_step < 0 or step < stop_step:
        return base_alpha
    if cooldown_steps <= 0:
        return 1.0
    progress = min(1.0, (step - stop_step) / cooldown_steps)
    return base_alpha + (1.0 - base_alpha) * progress


def compute_learning_rate(
    it: int,
    max_lr: float,
    min_lr: float,
    schedule: str,
    warmup_steps: int,
    lr_decay_iters: int,
    kd_high_lr_mult: float,
    kd_high_lr_pct: float,
    kd_drop_pct: float,
) -> float:
    if schedule == "constant":
        return max_lr

    boosted_peak_lr = max_lr * kd_high_lr_mult if schedule == "kd_boosted" else max_lr

    if warmup_steps > 0 and it < warmup_steps:
        return boosted_peak_lr * (it + 1) / (warmup_steps + 1)

    if it > lr_decay_iters:
        return min_lr

    denom = max(1, lr_decay_iters - warmup_steps)
    decay_ratio = (it - warmup_steps) / denom
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    if schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    if schedule == "inverse_sqrt":
        x = max(0, it - warmup_steps)
        max_x = max(1, lr_decay_iters - warmup_steps)
        min_to_peak_ratio_sq = (min_lr / max_lr) ** 2
        k = (min_to_peak_ratio_sq * max_x) / max(1e-12, (1.0 - min_to_peak_ratio_sq))
        lr = max_lr * math.sqrt(k / (x + k))
        return max(min_lr, min(lr, max_lr))
    if schedule == "kd_boosted":
        decay_span = max(1, lr_decay_iters - warmup_steps)
        high_steps = int(round(decay_span * kd_high_lr_pct))
        high_steps = max(0, min(high_steps, max(0, decay_span - 2)))
        drop_steps = int(round(decay_span * kd_drop_pct))
        drop_steps = max(1, min(drop_steps, max(1, decay_span - high_steps - 1)))
        high_end = warmup_steps + high_steps
        drop_end = min(lr_decay_iters - 1, high_end + drop_steps)

        if it <= high_end:
            return boosted_peak_lr
        if it <= drop_end:
            drop_ratio = (it - high_end) / max(1, drop_end - high_end)
            return boosted_peak_lr + (max_lr - boosted_peak_lr) * drop_ratio

        tail_ratio = (it - drop_end) / max(1, lr_decay_iters - drop_end)
        tail_ratio = min(max(tail_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * tail_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    raise ValueError(f"Unsupported lr schedule: {schedule}")


def build_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[TinyGPT, Dict[str, int]]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "args" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Checkpoint is missing required keys: 'args' and 'model'.")

    ckpt_args = checkpoint["args"]
    cfg = GPTConfig(
        vocab_size=int(ckpt_args["vocab_size"]),
        n_layer=int(ckpt_args["n_layer"]),
        n_head=int(ckpt_args["n_head"]),
        n_embd=int(ckpt_args["n_embd"]),
    )
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, {
        "vocab_size": cfg.vocab_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
    }


def setup_csv_logger(log_path: str) -> Tuple[str, csv.DictWriter, object]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    csv_file = open(log_path, "w", newline="")
    fieldnames = [
        "step",
        "num_iterations",
        "mode",
        "lr",
        "train_total_loss",
        "train_hard_loss",
        "train_soft_loss",
        "distill_alpha",
        "val_loss",
        "teacher_val_loss",
        "distill_stop_triggered",
        "distill_stop_triggered_step",
        "step_ms",
        "avg_ms",
        "tokens_per_sec",
        "timestamp_utc",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    return log_path, writer, csv_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mac-friendly tiny training runner for modded-nanogpt.")
    parser.add_argument(
        "--train-files",
        type=str,
        default="data/fineweb10B/fineweb_train_000001.bin",
        help="Glob for training .bin shard(s).",
    )
    parser.add_argument(
        "--val-files",
        type=str,
        default="data/fineweb10B/fineweb_val_000000.bin",
        help="Glob for validation .bin shard(s).",
    )
    parser.add_argument("--num-iterations", type=int, default=12, help="Number of train iterations.")
    parser.add_argument("--val-loss-every", type=int, default=5, help="Run validation every N steps.")
    parser.add_argument("--batch-size-tokens", type=int, default=4096, help="Total tokens per train step.")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument("--eval-steps", type=int, default=32, help="Number of batches per validation pass.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="fixed",
        choices=["fixed", "random"],
        help="Validation mode: fixed replay batches (stable curves) or fresh random batches (noisy).",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device target.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer blocks.")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Vocabulary size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Legacy alias for --max-lr. If --max-lr is unset, this value is used.",
    )
    parser.add_argument("--max-lr", type=float, default=None, help="Peak learning rate.")
    parser.add_argument("--min-lr", type=float, default=None, help="Minimum learning rate. Defaults to max_lr/10.")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["constant", "cosine", "inverse_sqrt", "kd_boosted"],
        help="Learning-rate schedule mode.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Linear warmup steps. Defaults to ceil(0.10 * num_iterations).",
    )
    parser.add_argument(
        "--lr-decay-iters",
        type=int,
        default=None,
        help="Cosine decay horizon. Defaults to num_iterations.",
    )
    parser.add_argument(
        "--kd-high-lr-mult",
        type=float,
        default=1.0,
        help="For --lr-schedule=kd_boosted: multiplier applied to max_lr during early KD phase.",
    )
    parser.add_argument(
        "--kd-high-lr-pct",
        type=float,
        default=0.20,
        help="For --lr-schedule=kd_boosted: fraction of decay horizon spent at boosted LR.",
    )
    parser.add_argument(
        "--kd-drop-pct",
        type=float,
        default=0.10,
        help="For --lr-schedule=kd_boosted: fraction of decay horizon used for rapid LR drop to max_lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--log-file", type=str, default="", help="Optional CSV path for train/val metrics.")
    parser.add_argument("--distill-enabled", action="store_true", help="Enable teacher-student distillation.")
    parser.add_argument("--teacher-checkpoint", type=str, default="", help="Teacher checkpoint path.")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight for hard CE loss in distillation.")
    parser.add_argument("--distill-temperature", type=float, default=2.0, help="Temperature for KD logits.")
    parser.add_argument(
        "--distill-stop-mode",
        type=str,
        default="val_margin",
        choices=["none", "fixed", "val_margin"],
        help="How to phase out distillation.",
    )
    parser.add_argument(
        "--distill-stop-margin",
        type=float,
        default=0.05,
        help="For val_margin mode: trigger stop when student_val <= teacher_val + margin.",
    )
    parser.add_argument(
        "--distill-stop-min-step",
        type=int,
        default=0,
        help="For val_margin mode: earliest step that can trigger KD stop.",
    )
    parser.add_argument("--distill-stop-step", type=int, default=-1, help="Step to start phasing distillation out.")
    parser.add_argument("--distill-cooldown-steps", type=int, default=0, help="Steps to anneal alpha toward CE-only.")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save final checkpoint to logs/mac/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_iterations < 1:
        raise ValueError("--num-iterations must be >= 1.")
    if args.val_loss_every < 1:
        raise ValueError("--val-loss-every must be >= 1.")
    if args.eval_steps < 1:
        raise ValueError("--eval-steps must be >= 1.")
    if args.batch_size_tokens < args.max_seq_len:
        raise ValueError("--batch-size-tokens must be >= --max-seq-len.")
    if args.n_layer < 1:
        raise ValueError("--n-layer must be >= 1.")
    if args.n_head < 1:
        raise ValueError("--n-head must be >= 1.")
    if args.n_embd < 8:
        raise ValueError("--n-embd must be >= 8.")
    if args.vocab_size < 128:
        raise ValueError("--vocab-size must be >= 128.")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head.")
    default_max_lr = 3e-4
    max_lr = args.max_lr if args.max_lr is not None else (
        args.learning_rate if args.learning_rate is not None else default_max_lr
    )
    min_lr = args.min_lr if args.min_lr is not None else (max_lr / 10.0)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else math.ceil(0.10 * args.num_iterations)
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.num_iterations

    if max_lr <= 0.0:
        raise ValueError("--max-lr/--learning-rate must be > 0.")
    if min_lr <= 0.0:
        raise ValueError("--min-lr must be > 0.")
    if min_lr > max_lr:
        raise ValueError("--min-lr must be <= --max-lr.")
    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0.")
    if lr_decay_iters < 1:
        raise ValueError("--lr-decay-iters must be >= 1.")
    if warmup_steps >= lr_decay_iters:
        raise ValueError("--warmup-steps must be < --lr-decay-iters.")
    if args.kd_high_lr_mult <= 0.0:
        raise ValueError("--kd-high-lr-mult must be > 0.")
    if not (0.0 <= args.kd_high_lr_pct < 1.0):
        raise ValueError("--kd-high-lr-pct must be in [0, 1).")
    if not (0.0 <= args.kd_drop_pct < 1.0):
        raise ValueError("--kd-drop-pct must be in [0, 1).")
    if args.kd_high_lr_pct + args.kd_drop_pct >= 1.0:
        raise ValueError("--kd-high-lr-pct + --kd-drop-pct must be < 1.0.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0.")
    if not (0.0 <= args.beta1 < 1.0):
        raise ValueError("--beta1 must be in [0, 1).")
    if not (0.0 <= args.beta2 < 1.0):
        raise ValueError("--beta2 must be in [0, 1).")
    if not (0.0 <= args.distill_alpha <= 1.0):
        raise ValueError("--distill-alpha must be in [0, 1].")
    if args.distill_temperature <= 0.0:
        raise ValueError("--distill-temperature must be > 0.")
    if args.distill_stop_margin < 0.0:
        raise ValueError("--distill-stop-margin must be >= 0.")
    if args.distill_stop_min_step < 0:
        raise ValueError("--distill-stop-min-step must be >= 0.")
    if args.distill_enabled and not args.teacher_checkpoint:
        raise ValueError("--teacher-checkpoint is required when --distill-enabled is set.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    batch_size_seqs = max(1, args.batch_size_tokens // args.max_seq_len)
    effective_tokens = batch_size_seqs * args.max_seq_len

    train_shards = resolve_shards(args.train_files)
    val_shards = resolve_shards(args.val_files)
    train_sampler = BatchSampler(train_shards, args.max_seq_len, batch_size_seqs, seed=args.seed)
    val_sampler = BatchSampler(val_shards, args.max_seq_len, batch_size_seqs, seed=args.seed + 1)
    fixed_val_batches = None
    if args.eval_mode == "fixed":
        fixed_val_batches = build_fixed_eval_batches(val_sampler, args.eval_steps)

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = TinyGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    teacher_model: Optional[TinyGPT] = None
    if args.distill_enabled:
        teacher_model, teacher_cfg = build_model_from_checkpoint(args.teacher_checkpoint, device)
        if teacher_cfg["vocab_size"] != cfg.vocab_size:
            raise ValueError("Teacher and student vocab sizes must match for logits distillation.")

    log_path = args.log_file
    if not log_path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("logs", "mac", f"train_log_{timestamp}.csv")
    _, csv_writer, csv_file = setup_csv_logger(log_path)

    print("=== train_gpt_mac startup ===")
    print(f"device={device.type}")
    print(f"seed={args.seed}")
    print(f"num_iterations={args.num_iterations}")
    print(f"val_loss_every={args.val_loss_every}")
    print(f"eval_mode={args.eval_mode}")
    print(f"eval_steps={args.eval_steps}")
    print(f"max_seq_len={args.max_seq_len}")
    print(f"batch_size_tokens={args.batch_size_tokens}")
    print(f"effective_batch_size_seqs={batch_size_seqs}")
    print(f"effective_tokens_per_step={effective_tokens}")
    print(f"model_n_layer={cfg.n_layer}")
    print(f"model_n_head={cfg.n_head}")
    print(f"model_n_embd={cfg.n_embd}")
    print(f"model_vocab_size={cfg.vocab_size}")
    print(f"max_lr={max_lr}")
    print(f"min_lr={min_lr}")
    print(f"lr_schedule={args.lr_schedule}")
    print(f"warmup_steps={warmup_steps}")
    print(f"lr_decay_iters={lr_decay_iters}")
    if args.lr_schedule == "kd_boosted":
        print(f"kd_high_lr_mult={args.kd_high_lr_mult}")
        print(f"kd_high_lr_pct={args.kd_high_lr_pct}")
        print(f"kd_drop_pct={args.kd_drop_pct}")
    print(f"weight_decay={args.weight_decay}")
    print(f"beta1={args.beta1}")
    print(f"beta2={args.beta2}")
    print(f"distill_enabled={args.distill_enabled}")
    if args.distill_enabled:
        print(f"teacher_checkpoint={args.teacher_checkpoint}")
        print(f"distill_alpha={args.distill_alpha}")
        print(f"distill_temperature={args.distill_temperature}")
        print(f"distill_stop_mode={args.distill_stop_mode}")
        print(f"distill_stop_margin={args.distill_stop_margin}")
        print(f"distill_stop_min_step={args.distill_stop_min_step}")
        print(f"distill_stop_step={args.distill_stop_step}")
        print(f"distill_cooldown_steps={args.distill_cooldown_steps}")
    print(f"log_file={log_path}")
    print(f"train_files={len(train_shards)}")
    print(f"val_files={len(val_shards)}")
    print(f"train_first={train_shards[0].path}")
    print(f"val_first={val_shards[0].path}")
    print(f"train_first_tokens={train_shards[0].num_tokens}")
    print(f"val_first_tokens={val_shards[0].num_tokens}")
    print(f"model_params={count_parameters(model):,}")

    total_ms = 0.0
    total_tokens = 0
    distill_stop_triggered_step: Optional[int] = None
    if args.distill_enabled and args.distill_stop_mode == "fixed" and args.distill_stop_step >= 0:
        distill_stop_triggered_step = args.distill_stop_step

    for step in range(1, args.num_iterations + 1):
        it = step - 1
        lr = compute_learning_rate(
            it=it,
            max_lr=max_lr,
            min_lr=min_lr,
            schedule=args.lr_schedule,
            warmup_steps=warmup_steps,
            lr_decay_iters=lr_decay_iters,
            kd_high_lr_mult=args.kd_high_lr_mult,
            kd_high_lr_pct=args.kd_high_lr_pct,
            kd_drop_pct=args.kd_drop_pct,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = train_sampler.sample()
        x = x.to(device)
        y = y.to(device)

        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits, hard_loss = model(x, y)
        soft_loss = None
        alpha = 1.0
        loss = hard_loss

        if args.distill_enabled:
            with torch.no_grad():
                teacher_logits, _ = teacher_model(x, None)
            temperature = args.distill_temperature
            student_log_probs = F.log_softmax(logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            # Token-level KL (mean over B,T) keeps KD magnitude comparable to CE.
            soft_loss = (
                F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean() * (temperature ** 2)
            )
            if args.distill_stop_mode == "none":
                alpha = args.distill_alpha
            else:
                stop_step = args.distill_stop_step if args.distill_stop_mode == "fixed" else (
                    distill_stop_triggered_step if distill_stop_triggered_step is not None else -1
                )
                alpha = compute_distill_alpha(
                    step=step,
                    base_alpha=args.distill_alpha,
                    stop_step=stop_step,
                    cooldown_steps=args.distill_cooldown_steps,
                )
            loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync_device(device)
        step_ms = (time.perf_counter() - step_start) * 1000.0

        total_ms += step_ms
        total_tokens += int(x.numel())
        avg_ms = total_ms / step
        toks_per_sec = total_tokens / (total_ms / 1000.0)
        hard_item = float(hard_loss.item())
        soft_item = float(soft_loss.item()) if soft_loss is not None else float("nan")
        total_item = float(loss.item())
        val_loss = None
        teacher_val_loss = None

        msg = (
            f"step={step}/{args.num_iterations} "
            f"lr={lr:.6g} "
            f"train_total_loss={total_item:.4f} "
            f"train_hard_loss={hard_item:.4f} "
            f"{(f'train_soft_loss={soft_item:.4f} ' if args.distill_enabled else '')}"
            f"{(f'distill_alpha={alpha:.4f} ' if args.distill_enabled else '')}"
            f"step_ms={step_ms:.2f} "
            f"avg_ms={avg_ms:.2f} "
            f"tokens_per_sec={toks_per_sec:.2f}"
        )

        if step % args.val_loss_every == 0 or step == args.num_iterations:
            val_loss = float(
                evaluate(
                    model=model,
                    device=device,
                    eval_steps=args.eval_steps,
                    sampler=(None if fixed_val_batches is not None else val_sampler),
                    fixed_batches=fixed_val_batches,
                )
            )
            msg = f"{msg} val_loss={val_loss:.4f}"
            if args.distill_enabled:
                teacher_val_loss = float(
                    evaluate(
                        model=teacher_model,
                        device=device,
                        eval_steps=args.eval_steps,
                        sampler=(None if fixed_val_batches is not None else val_sampler),
                        fixed_batches=fixed_val_batches,
                    )
                )
                msg = f"{msg} teacher_val_loss={teacher_val_loss:.4f}"
                if (
                    args.distill_stop_mode == "val_margin"
                    and distill_stop_triggered_step is None
                    and step >= args.distill_stop_min_step
                    and val_loss <= teacher_val_loss + args.distill_stop_margin
                ):
                    distill_stop_triggered_step = step
                    msg = f"{msg} distill_stop_triggered_step={distill_stop_triggered_step}"

        distill_stop_triggered = (
            distill_stop_triggered_step is not None and step >= distill_stop_triggered_step
        )

        print(msg)
        csv_writer.writerow(
            {
                "step": step,
                "num_iterations": args.num_iterations,
                "mode": "distill" if args.distill_enabled else "train",
                "lr": f"{lr:.8f}",
                "train_total_loss": f"{total_item:.6f}",
                "train_hard_loss": f"{hard_item:.6f}",
                "train_soft_loss": (f"{soft_item:.6f}" if args.distill_enabled else ""),
                "distill_alpha": (f"{alpha:.6f}" if args.distill_enabled else ""),
                "val_loss": (f"{val_loss:.6f}" if val_loss is not None else ""),
                "teacher_val_loss": (f"{teacher_val_loss:.6f}" if teacher_val_loss is not None else ""),
                "distill_stop_triggered": (str(distill_stop_triggered) if args.distill_enabled else ""),
                "distill_stop_triggered_step": (
                    str(distill_stop_triggered_step) if args.distill_enabled and distill_stop_triggered_step else ""
                ),
                "step_ms": f"{step_ms:.6f}",
                "avg_ms": f"{avg_ms:.6f}",
                "tokens_per_sec": f"{toks_per_sec:.6f}",
                "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
        csv_file.flush()

    print("=== run complete ===")
    print(f"final_avg_ms_per_step={total_ms / args.num_iterations:.2f}")
    print(f"final_tokens_per_sec={total_tokens / (total_ms / 1000.0):.2f}")

    if args.save_checkpoint:
        os.makedirs("logs/mac", exist_ok=True)
        ckpt_path = os.path.join("logs", "mac", f"state_step{args.num_iterations:06d}.pt")
        torch.save(
            {
                "step": args.num_iterations,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"checkpoint_saved={ckpt_path}")
    csv_file.close()


if __name__ == "__main__":
    main()
