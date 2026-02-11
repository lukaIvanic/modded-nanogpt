#!/usr/bin/env python3
"""Run the L07 teacher + two-stage KD sweep campaign on train_gpt_mac.py."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


TRAIN_SCRIPT = "train_gpt_mac.py"


@dataclass
class RunResult:
    run_id: str
    label: str
    csv_path: Path
    out_log_path: Path
    exit_code: int
    wall_seconds: float
    final_step: int
    final_train_loss: float
    final_hard_loss: float
    final_soft_loss: float
    final_alpha: float
    final_val_loss: float
    best_val_step: int
    best_val_loss: float
    final_avg_ms: float
    final_tokens_per_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full L07 KD campaign.")
    parser.add_argument("--workdir", type=Path, default=Path("."), help="modded-nanogpt root directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed-teacher", type=int, default=44)
    parser.add_argument("--seed-student", type=int, default=45)
    parser.add_argument("--batch-size-tokens", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--val-loss-every-short", type=int, default=15)
    parser.add_argument("--val-loss-every-long", type=int, default=155)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to logs/mac/kd_campaign_<timestamp>/",
    )
    return parser.parse_args()


def parse_train_csv(path: Path) -> Tuple[List[dict], List[dict]]:
    rows = [r for r in csv.DictReader(path.open()) if r.get("mode") in {"train", "distill"}]
    if not rows:
        raise ValueError(f"No train/distill rows in {path}")
    val_rows = [r for r in rows if r.get("val_loss")]
    return rows, val_rows


def to_float(value: str, default: float = float("nan")) -> float:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return float(value)


def run_train(
    workdir: Path,
    out_dir: Path,
    run_id: str,
    label: str,
    args: Dict[str, object],
) -> RunResult:
    csv_path = out_dir / f"{run_id}.csv"
    out_log_path = out_dir / f"{run_id}.out.log"

    cmd: List[str] = [".venv/bin/python", TRAIN_SCRIPT]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    cmd.extend(["--log-file", str(csv_path)])

    print(f"[run] {run_id} :: {label}", flush=True)
    t0 = time.perf_counter()
    with out_log_path.open("w") as out_f:
        proc = subprocess.run(cmd, cwd=workdir, stdout=out_f, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({run_id}) exit={proc.returncode}. See {out_log_path}")

    rows, val_rows = parse_train_csv(csv_path)
    last = rows[-1]
    best = min(val_rows, key=lambda r: float(r["val_loss"])) if val_rows else last

    result = RunResult(
        run_id=run_id,
        label=label,
        csv_path=csv_path,
        out_log_path=out_log_path,
        exit_code=proc.returncode,
        wall_seconds=wall,
        final_step=int(last["step"]),
        final_train_loss=float(last["train_total_loss"]),
        final_hard_loss=to_float(last.get("train_hard_loss", "")),
        final_soft_loss=to_float(last.get("train_soft_loss", "")),
        final_alpha=to_float(last.get("distill_alpha", "")),
        final_val_loss=to_float(last.get("val_loss", "")),
        best_val_step=int(best["step"]) if best.get("step") else int(last["step"]),
        best_val_loss=to_float(best.get("val_loss", "")),
        final_avg_ms=float(last["avg_ms"]),
        final_tokens_per_sec=float(last["tokens_per_sec"]),
    )
    print(
        f"[done] {run_id} val={result.final_val_loss:.6f} "
        f"avg_ms={result.final_avg_ms:.3f} wall={result.wall_seconds:.2f}s",
        flush=True,
    )
    return result


def write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_to_row(result: RunResult, extra: Dict[str, object]) -> Dict[str, object]:
    row = {
        "run_id": result.run_id,
        "label": result.label,
        "csv_path": str(result.csv_path),
        "out_log_path": str(result.out_log_path),
        "exit_code": result.exit_code,
        "wall_seconds": f"{result.wall_seconds:.3f}",
        "final_step": result.final_step,
        "final_train_loss": f"{result.final_train_loss:.6f}",
        "final_hard_loss": f"{result.final_hard_loss:.6f}",
        "final_soft_loss": f"{result.final_soft_loss:.6f}",
        "final_alpha": f"{result.final_alpha:.6f}",
        "final_val_loss": f"{result.final_val_loss:.6f}",
        "best_val_step": result.best_val_step,
        "best_val_loss": f"{result.best_val_loss:.6f}",
        "final_avg_ms": f"{result.final_avg_ms:.6f}",
        "final_tokens_per_sec": f"{result.final_tokens_per_sec:.6f}",
    }
    row.update(extra)
    return row


def plot_sweep(summary_rows: Sequence[Dict[str, object]], out_train: Path, out_val: Path, title_prefix: str) -> None:
    sorted_rows = sorted(summary_rows, key=lambda r: float(r["final_val_loss"]))

    plt.figure(figsize=(13, 7))
    for idx, row in enumerate(sorted_rows):
        csv_path = Path(str(row["csv_path"]))
        train_rows, _ = parse_train_csv(csv_path)
        x = [int(r["step"]) for r in train_rows]
        y = [float(r["train_hard_loss"]) for r in train_rows]
        lw = 2.5 if idx == 0 else 1.3
        alpha = 1.0 if idx == 0 else 0.8
        label = f"{row['sheet_id']} (val@{train_rows[-1]['step']}={float(row['final_val_loss']):.3f})"
        plt.plot(x, y, linewidth=lw, alpha=alpha, label=label)
    plt.title(f"{title_prefix}: Train Hard Loss vs Steps")
    plt.xlabel("steps")
    plt.ylabel("train loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_train, dpi=170)
    plt.close()

    plt.figure(figsize=(13, 7))
    for idx, row in enumerate(sorted_rows):
        csv_path = Path(str(row["csv_path"]))
        _, val_rows = parse_train_csv(csv_path)
        x = [int(r["step"]) for r in val_rows]
        y = [float(r["val_loss"]) for r in val_rows]
        lw = 2.7 if idx == 0 else 1.4
        alpha = 1.0 if idx == 0 else 0.85
        label = f"{row['sheet_id']} (val@{val_rows[-1]['step']}={float(row['final_val_loss']):.3f})"
        plt.plot(x, y, marker="o", markersize=3.3, linewidth=lw, alpha=alpha, label=label)
    plt.title(f"{title_prefix}: Validation Loss vs Steps")
    plt.xlabel("steps")
    plt.ylabel("validation loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_val, dpi=170)
    plt.close()


def plot_final_compare(
    teacher_csv: Path,
    kd1555_csv: Path,
    kd1000_csv: Path,
    out_train: Path,
    out_val: Path,
) -> None:
    runs = [
        ("CE Baseline 1555", teacher_csv, "#1f77b4"),
        ("KD 1555", kd1555_csv, "#d62728"),
        ("KD 1000", kd1000_csv, "#2ca02c"),
    ]

    plt.figure(figsize=(11.5, 6.5))
    for label, csv_path, color in runs:
        rows, _ = parse_train_csv(csv_path)
        x = [int(r["step"]) for r in rows]
        y = [float(r["train_hard_loss"]) for r in rows]
        plt.plot(x, y, linewidth=2.2, color=color, label=label)
    plt.title("Final Comparison: Train Hard Loss vs Steps")
    plt.xlabel("steps")
    plt.ylabel("train loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_train, dpi=170)
    plt.close()

    plt.figure(figsize=(11.5, 6.5))
    for label, csv_path, color in runs:
        _, val_rows = parse_train_csv(csv_path)
        x = [int(r["step"]) for r in val_rows]
        y = [float(r["val_loss"]) for r in val_rows]
        plt.plot(x, y, linewidth=2.2, marker="o", markersize=4, color=color, label=label)
    plt.title("Final Comparison: Validation Loss vs Steps")
    plt.xlabel("steps")
    plt.ylabel("validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_val, dpi=170)
    plt.close()


def refined_temps(base_t: float) -> List[float]:
    raw = [max(1.0, base_t - 0.4), max(1.0, base_t - 0.2), base_t, base_t + 0.2]
    out: List[float] = []
    for value in raw:
        candidate = round(value, 3)
        while candidate in out:
            candidate = round(candidate + 0.2, 3)
        out.append(candidate)
    return out


def sustained_within_margin(val_rows: Sequence[dict], threshold: float) -> bool:
    count = 0
    for row in val_rows:
        val = float(row["val_loss"])
        if val <= threshold:
            count += 1
            if count >= 2:
                return True
        else:
            count = 0
    return False


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.output_dir or (workdir / "logs" / "mac" / f"kd_campaign_{ts}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "train-files": "data/fineweb10B/fineweb_train_000001.bin",
        "val-files": "data/fineweb10B/fineweb_val_000000.bin",
        "batch-size-tokens": args.batch_size_tokens,
        "max-seq-len": args.max_seq_len,
        "eval-mode": "fixed",
        "eval-steps": args.eval_steps,
        "device": args.device,
        "n-layer": 2,
        "n-head": 2,
        "n-embd": 32,
        "max-lr": 1.1e-3,
        "min-lr": 1.1e-4,
        "lr-schedule": "cosine",
        "weight-decay": 0.03,
        "beta1": 0.9,
        "beta2": 0.95,
    }

    # Phase 2: CE teacher full 1555
    teacher_args = dict(common)
    teacher_args.update(
        {
            "seed": args.seed_teacher,
            "num-iterations": 1555,
            "val-loss-every": args.val_loss_every_long,
            "save-checkpoint": True,
        }
    )
    teacher = run_train(workdir, out_dir, "teacher_ce_l07_1555", "CE teacher L07 full run", teacher_args)
    teacher_ckpt_campaign = out_dir / "teacher_l07_1555.pt"
    teacher_ckpt_global = workdir / "logs" / "mac" / "teacher_l07_1555.pt"
    shutil.copy2(workdir / "logs" / "mac" / "state_step001555.pt", teacher_ckpt_campaign)
    shutil.copy2(workdir / "logs" / "mac" / "state_step001555.pt", teacher_ckpt_global)
    print(f"[checkpoint] teacher={teacher_ckpt_global}", flush=True)

    # Phase 3: KD sweep #1
    sweep1_rows: List[Dict[str, object]] = []
    sweep1_grid = list(
        itertools.product(
            [0.3, 0.5, 0.7],
            [1.0, 2.0],
            [0, 20],
        )
    )
    for idx, (alpha, temp, cooldown) in enumerate(sweep1_grid, start=1):
        sheet_id = f"S1_{idx:02d}"
        run_args = dict(common)
        run_args.update(
            {
                "seed": args.seed_student,
                "num-iterations": 155,
                "val-loss-every": args.val_loss_every_short,
                "distill-enabled": True,
                "teacher-checkpoint": str(teacher_ckpt_global),
                "distill-alpha": alpha,
                "distill-temperature": temp,
                "distill-stop-mode": "val_margin",
                "distill-stop-margin": 0.05,
                "distill-stop-min-step": 15,
                "distill-cooldown-steps": cooldown,
            }
        )
        result = run_train(workdir, out_dir, f"sweep1_{sheet_id}", f"Sweep1 {sheet_id}", run_args)
        sweep1_rows.append(
            result_to_row(
                result,
                {
                    "sheet_id": sheet_id,
                    "distill_alpha": alpha,
                    "distill_temperature": temp,
                    "distill_cooldown_steps": cooldown,
                },
            )
        )

    sweep1_rows.sort(key=lambda r: float(r["final_val_loss"]))
    write_summary(out_dir / "sweep1_summary_sorted.csv", sweep1_rows)
    plot_sweep(
        sweep1_rows,
        out_train=plots_dir / "sweep1_all_train_loss_vs_steps.png",
        out_val=plots_dir / "sweep1_all_val_loss_vs_steps.png",
        title_prefix="KD Sweep #1 (12 sheets)",
    )
    best1 = sweep1_rows[0]
    a_best = float(best1["distill_alpha"])
    t_best = float(best1["distill_temperature"])
    c_best = int(best1["distill_cooldown_steps"])
    print(f"[best sweep1] sheet={best1['sheet_id']} val={best1['final_val_loss']}", flush=True)

    # Phase 4: KD sweep #2 refined
    alpha_levels = sorted(
        {
            round(min(0.95, max(0.1, a_best - 0.15)), 3),
            round(min(0.95, max(0.1, a_best)), 3),
            round(min(0.95, max(0.1, a_best + 0.15)), 3),
        }
    )
    temp_levels = refined_temps(t_best)
    sweep2_rows: List[Dict[str, object]] = []
    sweep2_grid = list(itertools.product(alpha_levels, temp_levels))
    for idx, (alpha, temp) in enumerate(sweep2_grid, start=1):
        sheet_id = f"S2_{idx:02d}"
        run_args = dict(common)
        run_args.update(
            {
                "seed": args.seed_student,
                "num-iterations": 155,
                "val-loss-every": args.val_loss_every_short,
                "distill-enabled": True,
                "teacher-checkpoint": str(teacher_ckpt_global),
                "distill-alpha": alpha,
                "distill-temperature": temp,
                "distill-stop-mode": "val_margin",
                "distill-stop-margin": 0.05,
                "distill-stop-min-step": 15,
                "distill-cooldown-steps": c_best,
            }
        )
        result = run_train(workdir, out_dir, f"sweep2_{sheet_id}", f"Sweep2 {sheet_id}", run_args)
        sweep2_rows.append(
            result_to_row(
                result,
                {
                    "sheet_id": sheet_id,
                    "distill_alpha": alpha,
                    "distill_temperature": temp,
                    "distill_cooldown_steps": c_best,
                },
            )
        )

    sweep2_rows.sort(key=lambda r: float(r["final_val_loss"]))
    write_summary(out_dir / "sweep2_summary_sorted.csv", sweep2_rows)
    plot_sweep(
        sweep2_rows,
        out_train=plots_dir / "sweep2_all_train_loss_vs_steps.png",
        out_val=plots_dir / "sweep2_all_val_loss_vs_steps.png",
        title_prefix="KD Sweep #2 (12 sheets)",
    )
    best2 = sweep2_rows[0]
    print(f"[best sweep2] sheet={best2['sheet_id']} val={best2['final_val_loss']}", flush=True)

    best_alpha = float(best2["distill_alpha"])
    best_temp = float(best2["distill_temperature"])
    best_cooldown = int(best2["distill_cooldown_steps"])

    # Phase 5: full KD runs
    kd1555_args = dict(common)
    kd1555_args.update(
        {
            "seed": args.seed_student,
            "num-iterations": 1555,
            "val-loss-every": args.val_loss_every_long,
            "distill-enabled": True,
            "teacher-checkpoint": str(teacher_ckpt_global),
            "distill-alpha": best_alpha,
            "distill-temperature": best_temp,
            "distill-stop-mode": "val_margin",
            "distill-stop-margin": 0.05,
            "distill-stop-min-step": 15,
            "distill-cooldown-steps": best_cooldown,
            "save-checkpoint": True,
        }
    )
    kd1555 = run_train(workdir, out_dir, "kd_best_1555", "KD best full 1555", kd1555_args)

    kd1000_args = dict(common)
    kd1000_args.update(
        {
            "seed": args.seed_student,
            "num-iterations": 1000,
            "val-loss-every": 100,
            "distill-enabled": True,
            "teacher-checkpoint": str(teacher_ckpt_global),
            "distill-alpha": best_alpha,
            "distill-temperature": best_temp,
            "distill-stop-mode": "val_margin",
            "distill-stop-margin": 0.05,
            "distill-stop-min-step": 15,
            "distill-cooldown-steps": best_cooldown,
            "save-checkpoint": True,
        }
    )
    kd1000 = run_train(workdir, out_dir, "kd_best_1000", "KD best shortened 1000", kd1000_args)

    # Final comparison plots
    final_train_plot = plots_dir / "final_compare_train_loss_vs_steps.png"
    final_val_plot = plots_dir / "final_compare_val_loss_vs_steps.png"
    plot_final_compare(
        teacher_csv=teacher.csv_path,
        kd1555_csv=kd1555.csv_path,
        kd1000_csv=kd1000.csv_path,
        out_train=final_train_plot,
        out_val=final_val_plot,
    )

    # Success criterion: KD1000 within +0.05 of baseline final val for >=2 consecutive val checks.
    baseline_final_val = teacher.final_val_loss
    threshold = baseline_final_val + 0.05
    _, kd1000_val_rows = parse_train_csv(kd1000.csv_path)
    success = sustained_within_margin(kd1000_val_rows, threshold)

    # Final report
    final_rows = [
        result_to_row(teacher, {"role": "baseline_ce_1555"}),
        result_to_row(kd1555, {"role": "kd_1555"}),
        result_to_row(kd1000, {"role": "kd_1000"}),
    ]
    write_summary(out_dir / "final_runs_summary.csv", final_rows)

    report_path = out_dir / "REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# L07 KD Campaign Report",
                "",
                "## Core Metrics",
                f"- Baseline CE 1555 final val: {teacher.final_val_loss:.6f}",
                f"- KD 1555 final val: {kd1555.final_val_loss:.6f}",
                f"- KD 1000 final val: {kd1000.final_val_loss:.6f}",
                f"- Success threshold (baseline + 0.05): {threshold:.6f}",
                f"- KD1000 sustained success (2 consecutive val checks): {success}",
                "",
                "## Best Distillation Config (from Sweep #2)",
                f"- distill_alpha: {best_alpha}",
                f"- distill_temperature: {best_temp}",
                f"- distill_cooldown_steps: {best_cooldown}",
                "",
                "## Key Files",
                f"- Teacher checkpoint: {teacher_ckpt_global}",
                f"- Sweep #1 summary: {out_dir / 'sweep1_summary_sorted.csv'}",
                f"- Sweep #2 summary: {out_dir / 'sweep2_summary_sorted.csv'}",
                f"- Final runs summary: {out_dir / 'final_runs_summary.csv'}",
                f"- Sweep #1 train plot: {plots_dir / 'sweep1_all_train_loss_vs_steps.png'}",
                f"- Sweep #1 val plot: {plots_dir / 'sweep1_all_val_loss_vs_steps.png'}",
                f"- Sweep #2 train plot: {plots_dir / 'sweep2_all_train_loss_vs_steps.png'}",
                f"- Sweep #2 val plot: {plots_dir / 'sweep2_all_val_loss_vs_steps.png'}",
                f"- Final compare train plot: {final_train_plot}",
                f"- Final compare val plot: {final_val_plot}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== Campaign complete ===", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"success_kd1000={success}", flush=True)


if __name__ == "__main__":
    main()
