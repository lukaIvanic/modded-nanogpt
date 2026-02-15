#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"step:(\d+)/(\d+)\s+train_time:([0-9eE+\-.]+)ms\s+step_avg:([0-9eE+\-.]+)ms"
)


def extract_float(line: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}:([0-9eE+\-.]+)", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def p90(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, math.ceil(0.90 * len(s)) - 1))
    return s[idx]


@dataclass
class CaseSummary:
    case: str
    status: str
    exit_code: int | None
    last_step: int
    is_oom: bool
    oom_line: str
    median_step_ms_delta: float
    p90_step_ms_delta: float
    median_student_ms: float | None
    median_teacher_ms: float | None
    median_soft_ms: float | None
    median_backward_ms: float | None
    median_profile_sum_ms: float | None
    peak_alloc_mib: float | None
    peak_reserved_mib: float | None


def parse_case(case_dir: Path, drop_first_deltas: int) -> CaseSummary:
    status = "unknown"
    exit_code: int | None = None
    status_file = case_dir / "status.json"
    if status_file.exists():
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        status = str(payload.get("status", "unknown"))
        if isinstance(payload.get("exit_code"), int):
            exit_code = int(payload["exit_code"])

    driver_log = case_dir / "driver.log"
    oom_line = ""
    is_oom = False
    if driver_log.exists():
        for line in driver_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if re.search(r"out of memory|cuda error: out of memory|cublas_status_alloc_failed|cudnn_status_alloc_failed", line, re.I):
                is_oom = True
                oom_line = line.strip()
                break

    train_log = case_dir / "train.log"
    train_times: list[tuple[int, float]] = []
    student_ms: list[float] = []
    teacher_ms: list[float] = []
    soft_ms: list[float] = []
    backward_ms: list[float] = []
    profile_sum_ms: list[float] = []
    peak_alloc_mib: list[float] = []
    peak_reserved_mib: list[float] = []

    if train_log.exists():
        for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = TRAIN_RE.search(line)
            if not m or "train_hard_loss:" not in line:
                continue
            step = int(m.group(1))
            train_time = float(m.group(3))
            train_times.append((step, train_time))
            s = extract_float(line, "kd_prof_student_ms")
            t = extract_float(line, "kd_prof_teacher_ms")
            so = extract_float(line, "kd_prof_soft_ms")
            b = extract_float(line, "kd_prof_backward_ms")
            ps = extract_float(line, "kd_prof_step_profile_ms")
            pa = extract_float(line, "kd_prof_peak_alloc_mib")
            pr = extract_float(line, "kd_prof_peak_reserved_mib")
            if s is not None:
                student_ms.append(s)
            if t is not None:
                teacher_ms.append(t)
            if so is not None:
                soft_ms.append(so)
            if b is not None:
                backward_ms.append(b)
            if ps is not None:
                profile_sum_ms.append(ps)
            if pa is not None:
                peak_alloc_mib.append(pa)
            if pr is not None:
                peak_reserved_mib.append(pr)

    deltas: list[float] = []
    for i in range(1, len(train_times)):
        d = train_times[i][1] - train_times[i - 1][1]
        if math.isfinite(d) and d >= 0:
            deltas.append(d)
    steady = deltas[drop_first_deltas:] if len(deltas) > drop_first_deltas else []

    last_step = train_times[-1][0] if train_times else 0
    return CaseSummary(
        case=case_dir.name,
        status=status,
        exit_code=exit_code,
        last_step=last_step,
        is_oom=is_oom,
        oom_line=oom_line,
        median_step_ms_delta=median(steady),
        p90_step_ms_delta=p90(steady),
        median_student_ms=(median(student_ms[drop_first_deltas:]) if len(student_ms) > drop_first_deltas else None),
        median_teacher_ms=(median(teacher_ms[drop_first_deltas:]) if len(teacher_ms) > drop_first_deltas else None),
        median_soft_ms=(median(soft_ms[drop_first_deltas:]) if len(soft_ms) > drop_first_deltas else None),
        median_backward_ms=(median(backward_ms[drop_first_deltas:]) if len(backward_ms) > drop_first_deltas else None),
        median_profile_sum_ms=(median(profile_sum_ms[drop_first_deltas:]) if len(profile_sum_ms) > drop_first_deltas else None),
        peak_alloc_mib=(max(peak_alloc_mib) if peak_alloc_mib else None),
        peak_reserved_mib=(max(peak_reserved_mib) if peak_reserved_mib else None),
    )


def write_csv(path: Path, rows: list[CaseSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "status",
                "exit_code",
                "last_step",
                "is_oom",
                "oom_line",
                "median_step_ms_delta",
                "p90_step_ms_delta",
                "median_student_ms",
                "median_teacher_ms",
                "median_soft_ms",
                "median_backward_ms",
                "median_profile_sum_ms",
                "peak_alloc_mib",
                "peak_reserved_mib",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.case,
                    r.status,
                    "" if r.exit_code is None else r.exit_code,
                    r.last_step,
                    int(r.is_oom),
                    r.oom_line,
                    "" if not math.isfinite(r.median_step_ms_delta) else f"{r.median_step_ms_delta:.4f}",
                    "" if not math.isfinite(r.p90_step_ms_delta) else f"{r.p90_step_ms_delta:.4f}",
                    "" if r.median_student_ms is None else f"{r.median_student_ms:.4f}",
                    "" if r.median_teacher_ms is None else f"{r.median_teacher_ms:.4f}",
                    "" if r.median_soft_ms is None else f"{r.median_soft_ms:.4f}",
                    "" if r.median_backward_ms is None else f"{r.median_backward_ms:.4f}",
                    "" if r.median_profile_sum_ms is None else f"{r.median_profile_sum_ms:.4f}",
                    "" if r.peak_alloc_mib is None else f"{r.peak_alloc_mib:.4f}",
                    "" if r.peak_reserved_mib is None else f"{r.peak_reserved_mib:.4f}",
                ]
            )


def plot_step_delta(rows: list[CaseSummary], out: Path) -> None:
    labels = [r.case for r in rows if math.isfinite(r.median_step_ms_delta)]
    vals = [r.median_step_ms_delta for r in rows if math.isfinite(r.median_step_ms_delta)]
    if not labels:
        return
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("Median Step Delta (ms)")
    plt.title("KD Soft-Loss Debug: Median Step Delta by Case")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def parse_first_fit_ga(campaign_root: Path) -> str:
    p = campaign_root / "first_fit_ga.txt"
    if not p.exists():
        return "UNKNOWN"
    return p.read_text(encoding="utf-8").strip() or "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--drop-first-deltas", type=int, default=20)
    ap.add_argument("--steps", type=int, default=80)
    args = ap.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    cases_dir = campaign_root / "cases"
    if not cases_dir.exists():
        raise SystemExit(f"cases dir missing: {cases_dir}")

    rows: list[CaseSummary] = []
    for case_dir in sorted([p for p in cases_dir.iterdir() if p.is_dir()]):
        rows.append(parse_case(case_dir, args.drop_first_deltas))

    write_csv(campaign_root / "summary.csv", rows)
    plot_step_delta(rows, campaign_root / "step_delta_median.png")

    # Build report sections
    ga_rows = [r for r in rows if r.case.startswith("ga") and "_full_kd_unchunked" in r.case]
    decomp_rows = [r for r in rows if r.case.startswith("decomp_")]
    first_fit = parse_first_fit_ga(campaign_root)

    def find(name_prefix: str) -> CaseSummary | None:
        for r in decomp_rows:
            if r.case.startswith(name_prefix):
                return r
        return None

    ce = find("decomp_ce_only")
    tandem = find("decomp_tandem_infer_only")
    full_unch = find("decomp_full_kd_unchunked")
    full_chunk = find("decomp_full_kd_chunk1024")

    teacher_over = float("nan")
    soft_over = float("nan")
    chunk_impact = float("nan")
    if ce and tandem and math.isfinite(ce.median_step_ms_delta) and math.isfinite(tandem.median_step_ms_delta):
        teacher_over = tandem.median_step_ms_delta - ce.median_step_ms_delta
    if tandem and full_unch and math.isfinite(tandem.median_step_ms_delta) and math.isfinite(full_unch.median_step_ms_delta):
        soft_over = full_unch.median_step_ms_delta - tandem.median_step_ms_delta
    if full_unch and full_chunk and math.isfinite(full_unch.median_step_ms_delta) and math.isfinite(full_chunk.median_step_ms_delta):
        chunk_impact = full_chunk.median_step_ms_delta - full_unch.median_step_ms_delta

    decision = "UNCHUNKED_NOT_VIABLE_1GPU"
    if first_fit not in ("NONE", "UNKNOWN"):
        decision = f"UNCHUNKED_VIABLE_AT_GA={first_fit}"

    report_path = campaign_root / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# KD Soft-Loss Debug Report\n\n")
        f.write(f"- campaign_root: `{campaign_root}`\n")
        f.write(f"- drop_first_deltas: `{args.drop_first_deltas}`\n")
        f.write(f"- target_steps_per_case: `{args.steps}`\n")
        f.write(f"- first_fit_ga: `{first_fit}`\n")
        f.write(f"- decision: `{decision}`\n\n")

        f.write("## Unchunked Viability by GA\n\n")
        f.write("| case | status | last_step | oom | median_step_ms_delta | p90_step_ms_delta | peak_alloc_mib | peak_reserved_mib |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in ga_rows:
            med = "" if not math.isfinite(r.median_step_ms_delta) else f"{r.median_step_ms_delta:.1f}"
            p90v = "" if not math.isfinite(r.p90_step_ms_delta) else f"{r.p90_step_ms_delta:.1f}"
            pa = "" if r.peak_alloc_mib is None else f"{r.peak_alloc_mib:.1f}"
            pr = "" if r.peak_reserved_mib is None else f"{r.peak_reserved_mib:.1f}"
            f.write(
                f"| {r.case} | {r.status} | {r.last_step} | {int(r.is_oom)} | {med} | {p90v} | {pa} | {pr} |\n"
            )
        f.write("\n")

        f.write("## Component Timing Decomposition\n\n")
        f.write("| case | median_step_ms_delta | median_student_ms | median_teacher_ms | median_soft_ms | median_backward_ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in decomp_rows:
            med = "" if not math.isfinite(r.median_step_ms_delta) else f"{r.median_step_ms_delta:.1f}"
            s = "" if r.median_student_ms is None else f"{r.median_student_ms:.1f}"
            t = "" if r.median_teacher_ms is None else f"{r.median_teacher_ms:.1f}"
            so = "" if r.median_soft_ms is None else f"{r.median_soft_ms:.1f}"
            b = "" if r.median_backward_ms is None else f"{r.median_backward_ms:.1f}"
            f.write(f"| {r.case} | {med} | {s} | {t} | {so} | {b} |\n")
        f.write("\n")

        f.write("## Overhead Split (same GA)\n\n")
        f.write(f"- teacher_forward_overhead_ms: `{teacher_over:.1f}`\n" if math.isfinite(teacher_over) else "- teacher_forward_overhead_ms: ``\n")
        f.write(f"- softloss_overhead_ms_unchunked: `{soft_over:.1f}`\n" if math.isfinite(soft_over) else "- softloss_overhead_ms_unchunked: ``\n")
        f.write(f"- chunk1024_minus_unchunked_ms: `{chunk_impact:.1f}`\n" if math.isfinite(chunk_impact) else "- chunk1024_minus_unchunked_ms: ``\n")
        f.write("\n")

        f.write("## VRAM Pressure Profile\n\n")
        f.write("| case | peak_alloc_mib | peak_reserved_mib | oom |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            pa = "" if r.peak_alloc_mib is None else f"{r.peak_alloc_mib:.1f}"
            pr = "" if r.peak_reserved_mib is None else f"{r.peak_reserved_mib:.1f}"
            f.write(f"| {r.case} | {pa} | {pr} | {int(r.is_oom)} |\n")
        f.write("\n")

        f.write("## OOM Events\n\n")
        for r in rows:
            if r.is_oom:
                f.write(f"- {r.case}: `{r.oom_line}`\n")
        f.write("\n")

    print(f"wrote: {campaign_root / 'summary.csv'}")
    print(f"wrote: {campaign_root / 'report.md'}")
    print(f"wrote: {campaign_root / 'step_delta_median.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
