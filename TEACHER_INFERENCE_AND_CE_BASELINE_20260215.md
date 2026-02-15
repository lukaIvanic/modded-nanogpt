# Teacher Inference + CE Baseline (2026-02-15)

## Scope
- Host: `root@135.181.63.152` (single H100 80GB).
- Goal: measure teacher-only inference step speed and CE-only student step speed with compile and warmup enabled.
- Primary metric: `step_ms_delta = train_time(step_n) - train_time(step_n-1)`.
- Steady window: drop first `20` deltas.

## Model Context
- Student config (all runs): `layers=11 heads=6 head_dim=128 model_dim=768`.
- Student params: `543,818,890` (`543.819M`).
- Teacher benchmark config: random teacher model with same width/head dims but `layers=3`.
- Teacher params (bench run): `489,554,538` (`489.555M`).

## Teacher-Only Inference A/B
Benchmark mode used:
- `KD_ENABLED=1`
- `KD_TEACHER_BENCH_ONLY=1`
- `KD_RANDOM_MODEL_TEACHER=1`
- `KD_TEACHER_NUM_LAYERS=3`
- `KD_TEACHER_NUM_HEADS=6`
- `KD_TEACHER_HEAD_DIM=128`
- `KD_TEACHER_MODEL_DIM=768`
- `KD_COMPILE_TEACHER=1`
- `DISABLE_COMPILE=0`
- `SKIP_WARMUP=0`
- `NUM_SCHEDULED_ITERATIONS=120`
- `VAL_LOSS_EVERY=0`

Cases:
- `A_default`: `KD_TEACHER_FORCE_SDPA=0`
- `B_sdpa`: `KD_TEACHER_FORCE_SDPA=1`

Key runtime markers from trainer logs:
- `A_default`: `Teacher compile enabled=True`
- `B_sdpa`: `Teacher compile enabled=False`

Results:

| case | median_step_ms_delta | p90_step_ms_delta | latest_step_avg_ms |
|---|---:|---:|---:|
| `A_default` | `67.0` | `101.0` | `66.88` |
| `B_sdpa` | `424.0` | `728.0` | `490.62` |

Winner:
- `winner=A_default median_ms=67.00 p90_ms=101.00`

Artifacts:
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_infer_bench/20260215T133539Z_randL3H6D128/teacher_infer_ab_summary.csv`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_infer_bench/20260215T133539Z_randL3H6D128/teacher_infer_ab_report.md`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_infer_bench/20260215T133539Z_randL3H6D128/teacher_infer_ab_step_delta.png`
- Remote trainer logs:
  - `/data/nanogpt-speedrun/modded-nanogpt/logs/teacher_infer_A_default_20260215T133539Z.txt`
  - `/data/nanogpt-speedrun/modded-nanogpt/logs/teacher_infer_B_sdpa_20260215T133539Z.txt`

## CE-Only Student Baseline
Benchmark mode used:
- `KD_ENABLED=0`
- `DISABLE_COMPILE=0`
- `SKIP_WARMUP=0`
- `NUM_SCHEDULED_ITERATIONS=120`
- `VAL_LOSS_EVERY=0`

Result:

| case | median_step_ms_delta | p90_step_ms_delta | latest_step_avg_ms |
|---|---:|---:|---:|
| `ce_student_only` | `476.0` | `679.0` | `525.38` |

Artifacts:
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/ce_student_bench/20260215T134535Z/ce_student_summary.csv`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/ce_student_bench/20260215T134535Z/ce_student_report.md`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/ce_student_bench/20260215T134535Z/ce_student_step_delta.png`
- Remote trainer log:
  - `/data/nanogpt-speedrun/modded-nanogpt/logs/ce_student_bench_20260215T134535Z.txt`

## Quick Baseline Takeaway
- Teacher-only inference (`A_default`) is much faster than CE student training in this setup.
- Relative speed (median): `476 / 67 = ~7.1x` (student CE vs teacher-only inference).

## KD Soft-Loss Debug Snapshot (2026-02-15)

### Exact Findings
- **Unchunked exact KL probe** (`KD_SOFT_LOSS_TOKEN_CHUNK=0`):
  - artifacts: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_unchunked_probe/20260215T143633Z`
  - OOM during warmup soft-loss path.
  - error: tried `9.21 GiB`, free `3.32 GiB`, shortfall `~5.89 GiB`.
  - no runtime `step:` lines, so no stable step-ms from that run.
- **Chunk=8192 exact KL probe**:
  - artifacts: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_chunk_probe/20260215T144819Z`
  - OOM during warmup.
  - error: tried `1.54 GiB`, free `~0.30 GiB` (`298.12 MiB`).
- **Chunk=1024 exact KL probe** (random teacher model, same teacher architecture):
  - artifacts: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_chunk_probe/20260215T145059Z`
  - no OOM up to step `102`.
  - median step delta (drop first 20): `~3359 ms`, p90 `~6624 ms`.
  - median KD profile components:
    - student `~147.6 ms`
    - teacher `~61.8 ms`
    - soft `~454.0 ms`
    - backward `~2673.2 ms` (dominant)
  - median peak VRAM:
    - allocated `~56.3 GiB`
    - reserved `~78.3 GiB`

### Interpretation Note
- Slowdown in current full-KD exact path is dominated by backward under heavy memory pressure, not by teacher inference alone.
