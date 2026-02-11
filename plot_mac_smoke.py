#!/usr/bin/env python3
"""Create clickable PNG reports for train_gpt_mac smoke runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


CPU_RE = re.compile(r"([0-9.]+)% (user|sys|idle)")


def _to_float(value: str) -> float:
    if value is None:
        return math.nan
    value = value.strip()
    if not value:
        return math.nan
    return float(value)


def parse_train_csv(path: Path) -> Dict[str, object]:
    rows: List[dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no rows.")

    steps: List[int] = []
    train_total: List[float] = []
    train_hard: List[float] = []
    step_ms: List[float] = []
    avg_ms: List[float] = []
    tokens_per_sec: List[float] = []
    val_steps: List[int] = []
    val_loss: List[float] = []

    for row in rows:
        if row.get("mode") not in {"train", "distill"}:
            continue
        step = int(row["step"])
        steps.append(step)
        train_total.append(_to_float(row.get("train_total_loss", "")))
        train_hard.append(_to_float(row.get("train_hard_loss", "")))
        step_ms.append(_to_float(row.get("step_ms", "")))
        avg_ms.append(_to_float(row.get("avg_ms", "")))
        tokens_per_sec.append(_to_float(row.get("tokens_per_sec", "")))
        if row.get("val_loss", "").strip():
            val_steps.append(step)
            val_loss.append(float(row["val_loss"]))

    if not steps:
        raise ValueError(f"{path} has no train rows.")

    return {
        "steps": steps,
        "train_total": train_total,
        "train_hard": train_hard,
        "step_ms": step_ms,
        "avg_ms": avg_ms,
        "tokens_per_sec": tokens_per_sec,
        "val_steps": val_steps,
        "val_loss": val_loss,
    }


def parse_sys_csv(path: Path) -> Dict[str, List[float]]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    sample_idx: List[int] = []
    user_cpu: List[float] = []
    sys_cpu: List[float] = []
    idle_cpu: List[float] = []
    trainer_cpu: List[float] = []
    trainer_mem_pct: List[float] = []
    trainer_rss_mb: List[float] = []

    for idx, row in enumerate(rows, start=1):
        sample_idx.append(idx)
        usage = row.get("system_cpu_usage", "")
        parsed = {k: float(v) for v, k in CPU_RE.findall(usage)}
        user_cpu.append(parsed.get("user", math.nan))
        sys_cpu.append(parsed.get("sys", math.nan))
        idle_cpu.append(parsed.get("idle", math.nan))
        trainer_cpu.append(_to_float(row.get("trainer_cpu_percent", "")))
        trainer_mem_pct.append(_to_float(row.get("trainer_mem_percent", "")))
        rss_kb = _to_float(row.get("trainer_rss_kb", ""))
        trainer_rss_mb.append(rss_kb / 1024.0 if not math.isnan(rss_kb) else math.nan)

    return {
        "sample_idx": sample_idx,
        "user_cpu": user_cpu,
        "sys_cpu": sys_cpu,
        "idle_cpu": idle_cpu,
        "trainer_cpu": trainer_cpu,
        "trainer_mem_pct": trainer_mem_pct,
        "trainer_rss_mb": trainer_rss_mb,
    }


def parse_out_log(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    info: Dict[str, str] = {}
    interesting_keys = {
        "device",
        "model_params",
        "model_n_layer",
        "model_n_head",
        "model_n_embd",
        "batch_size_tokens",
        "max_seq_len",
        "num_iterations",
        "learning_rate",
        "max_lr",
        "min_lr",
        "lr_schedule",
        "weight_decay",
        "beta1",
        "beta2",
        "final_avg_ms_per_step",
        "final_tokens_per_sec",
    }
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in interesting_keys:
            info[key] = value
    return info


def run_label(path: Path, out_info: Dict[str, str]) -> str:
    stem = path.stem
    embd = out_info.get("model_n_embd")
    layer = out_info.get("model_n_layer")
    head = out_info.get("model_n_head")
    params = out_info.get("model_params")
    if embd and layer and head:
        label = f"{stem} (L{layer} H{head} D{embd})"
    else:
        label = stem
    if params:
        label += f" {params} params"
    return label


def make_dashboard(
    train_path: Path,
    data: Dict[str, object],
    sys_data: Dict[str, List[float]],
    out_info: Dict[str, str],
    output_path: Path,
) -> None:
    steps = data["steps"]
    train_total = data["train_total"]
    train_hard = data["train_hard"]
    val_steps = data["val_steps"]
    val_loss = data["val_loss"]
    step_ms = data["step_ms"]
    avg_ms = data["avg_ms"]
    tokens_per_sec = data["tokens_per_sec"]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(run_label(train_path, out_info), fontsize=13)

    ax = axs[0, 0]
    ax.plot(steps, train_hard, label="train hard", color="#2ca02c", linewidth=1.8)
    if val_steps:
        ax.scatter(val_steps, val_loss, label="val", color="#d62728", s=28, zorder=3)
    ax.set_title("Train Hard Loss vs Step")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axs[0, 1]
    ax.plot(steps, step_ms, label="step_ms", color="#9467bd", linewidth=1.4, alpha=0.75)
    ax.plot(steps, avg_ms, label="avg_ms", color="#ff7f0e", linewidth=2.0)
    ax.set_title("Step Time")
    ax.set_xlabel("step")
    ax.set_ylabel("ms")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axs[1, 0]
    ax.plot(steps, tokens_per_sec, label="tokens/sec", color="#17becf", linewidth=2.0)
    ax.set_title("Throughput")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens/sec")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axs[1, 1]
    if sys_data:
        x = sys_data["sample_idx"]
        ax.plot(x, sys_data["user_cpu"], label="system user%", color="#1f77b4", linewidth=1.4)
        ax.plot(x, sys_data["sys_cpu"], label="system sys%", color="#d62728", linewidth=1.4)
        ax.plot(x, sys_data["idle_cpu"], label="system idle%", color="#2ca02c", linewidth=1.4)
        ax2 = ax.twinx()
        ax2.plot(
            x,
            sys_data["trainer_cpu"],
            label="trainer cpu%",
            color="#9467bd",
            linewidth=1.5,
            linestyle="--",
        )
        ax2.set_ylabel("trainer cpu%")
        ax.set_title("System Load Samples")
        ax.set_xlabel("sample index (~2s interval)")
        ax.set_ylabel("system cpu%")
        ax.grid(alpha=0.25)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No system monitor CSV found", ha="center", va="center")
        ax.set_title("System Load Samples")
        ax.axis("off")

    summary_items = [
        f"final train hard loss: {train_hard[-1]:.4f}",
        f"final avg ms: {avg_ms[-1]:.2f}",
        f"final tok/s: {tokens_per_sec[-1]:.2f}",
    ]
    if val_loss:
        summary_items.insert(1, f"final val loss: {val_loss[-1]:.4f}")
    if out_info.get("device"):
        summary_items.append(f"device: {out_info['device']}")
    if out_info.get("batch_size_tokens") and out_info.get("max_seq_len"):
        summary_items.append(
            f"batch_size_tokens={out_info['batch_size_tokens']} max_seq_len={out_info['max_seq_len']}"
        )
    lr_value = out_info.get("max_lr", out_info.get("learning_rate"))
    if lr_value:
        summary_items.append(
            "opt="
            f"AdamW(lr={lr_value}, wd={out_info.get('weight_decay', '?')}, "
            f"betas=({out_info.get('beta1', '?')},{out_info.get('beta2', '?')}))"
        )
    if out_info.get("lr_schedule"):
        summary_items.append(f"lr_schedule={out_info['lr_schedule']}")
    fig.text(0.01, 0.01, " | ".join(summary_items), fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_comparison(
    train_paths: List[Path],
    runs: List[Dict[str, object]],
    labels: List[str],
    output_path: Path,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Mac Smoke Run Comparison", fontsize=14)

    for run, label in zip(runs, labels):
        axs[0].plot(run["steps"], run["train_hard"], label=label, linewidth=1.8)
        if run["val_steps"]:
            axs[0].scatter(run["val_steps"], run["val_loss"], s=20)
        axs[1].plot(run["steps"], run["avg_ms"], label=label, linewidth=1.8)
        axs[2].plot(run["steps"], run["tokens_per_sec"], label=label, linewidth=1.8)

    axs[0].set_title("Train Hard Loss vs Step")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("loss")
    axs[0].grid(alpha=0.25)

    axs[1].set_title("Avg Step Time")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("ms")
    axs[1].grid(alpha=0.25)

    axs[2].set_title("Throughput")
    axs[2].set_xlabel("step")
    axs[2].set_ylabel("tokens/sec")
    axs[2].grid(alpha=0.25)

    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    axs[2].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def latest_train_csv(log_dir: Path) -> Path:
    # Accept run files and skip *.sys.csv sidecars.
    candidates = [p for p in log_dir.glob("*.csv") if not p.name.endswith(".sys.csv")]
    if not candidates:
        raise FileNotFoundError(f"No train CSV found in {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train_gpt_mac smoke runs into PNG dashboards.")
    parser.add_argument(
        "--train-csv",
        nargs="+",
        default=[],
        help="One or more training CSV files. If omitted, --latest is used.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest training CSV from logs/mac when --train-csv is not set.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/mac"),
        help="Directory for auto-discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/mac/plots"),
        help="Directory for generated PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths: List[Path]
    if args.train_csv:
        train_paths = [Path(p) for p in args.train_csv]
    else:
        if not args.latest:
            args.latest = True
        train_paths = [latest_train_csv(args.log_dir)]

    runs: List[Dict[str, object]] = []
    labels: List[str] = []
    dashboard_paths: List[Path] = []
    for train_path in train_paths:
        train_path = train_path.resolve()
        base = train_path.with_suffix("")
        sys_path = Path(str(base) + ".sys.csv")
        out_log = Path(str(base) + ".out.log")
        train_data = parse_train_csv(train_path)
        sys_data = parse_sys_csv(sys_path) if sys_path.exists() else {}
        out_info = parse_out_log(out_log)
        label = run_label(train_path, out_info)
        runs.append(train_data)
        labels.append(label)
        dashboard_path = (args.output_dir / f"{train_path.stem}.dashboard.png").resolve()
        make_dashboard(train_path, train_data, sys_data, out_info, dashboard_path)
        dashboard_paths.append(dashboard_path)
        print(f"dashboard={dashboard_path}")

    if len(train_paths) > 1:
        cmp_name = "__vs__".join([p.stem for p in train_paths])
        cmp_path = (args.output_dir / f"{cmp_name}.comparison.png").resolve()
        make_comparison(train_paths, runs, labels, cmp_path)
        print(f"comparison={cmp_path}")


if __name__ == "__main__":
    main()
