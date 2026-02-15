# KD Hill-Climb Goals and Targets

> [!IMPORTANT]
> **North Star**
> Minimize student `steps_to_target` using logits KD, while keeping teacher pretraining cost very small.
> Target condition remains: student reaches `val_loss <= teacher_final_val` for **2 consecutive** validations.

> [!IMPORTANT]
> **New Hard Cost Constraint (Locked)**
> Teacher pretraining time budget is **<= 10%** of CE student full-run training time.
> Reference CE log:
> `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/remote_h100/server_logs/aadab377-e981-419d-907b-470c4e770f33.txt`
> CE train time at step 1555: `694015 ms` -> teacher budget: `69402 ms` (`69.402 s`).

> [!WARNING]
> **Disqualifiers**
> - Any candidate with projected teacher train time above `69402 ms` is rejected.
> - Any run without deterministic artifacts (`train.log`, summary CSV, plots, status) is rejected.
> - Historic ~163M teacher run is archival context only and is not used for final claims.

> [!TIP]
> **Hill-Climb Loop (Cost-Aware)**
> 1. Fix CE baseline runtime + target extraction.
> 2. Sweep tiny teacher candidates with kernel-compatible configs.
> 3. Keep only candidates satisfying runtime budget.
> 4. Pick best teacher by validation quality under budget.
> 5. Run KD sweeps/full runs versus CE and compare `steps_to_target`.

## Locked Constraints
- Student comparison regime: identical CE/KD semantics.
- Validation cadence: `VAL_LOSS_EVERY=25`.
- KD method: logits KD only, dynamic KD normalization ON.
- Teacher architecture no longer locked to 25M band; now locked to runtime budget.
- Teacher candidate policy: use kernel-compatible shapes (even heads, `head_dim % 4 == 0`), bigram minimized (`BIGRAM_VOCAB_SIZE=2`), tied embedding behavior unchanged unless explicitly modified.

> [!IMPORTANT]
> **KD Learning-Rate Policy (Locked From Paper + Prior Code)**
> - During active KD, student can safely run at higher LR than CE-only.
> - When KD contribution ends (teacher no longer clearly better), LR must drop quickly to avoid a low-loss floor caused by too-high LR.
> - Dynamic KD normalization stays mandatory so KD signal stays scale-matched to hard CE.

## KD LR Regime (Historical Prior, To Implement)
- Sources used:
  - `/Users/lukaivanic/projects/nanogpt-speedrun/self-distill-research/main_space/utils/hyperparameter_utils.py` (`scheduler_type="custom"`; high early LR then stepwise sharp drops)
  - `/Users/lukaivanic/projects/nanogpt-speedrun/self-distill-research/main_space/utils/train_utils.py` (dynamic soft/hard loss normalization)
  - `/Users/lukaivanic/projects/nanogpt-speedrun/DISTILLATION_RESEARCH_INTERVIEW.md` (KD useful while teacher is better; stop around +0.05 val margin)
- Canonical interpretation for this project:
  - `Phase A (KD-dominant)`: high LR while KD is active (boosted vs CE baseline).
  - `Phase B (handoff)`: sharp LR decay near KD stop to recover fine-grain optimization.
  - `Phase C (CE-finish)`: lower LR tail for final convergence.
- Legacy custom-schedule shape from prior project (12k-step reference):
  - `0-2000: 1e-3`, `2000-2600: 5e-4`, `2600-3200: 2e-4`, `3200-3600: 1e-4`, `3600-4200: 7e-5`,
    `4200-5200: 3e-5`, `5200-6000: 1e-5`, `6000-7000: 8e-6`, `7000-8000: 7e-6`, `8000-9000: 5e-6`,
    `9000-10500: 3e-6`, `10500+: 1e-6`.
- Fractional form to reuse on 155/1000/1555-step runs:
  - hold boosted LR for first `~15-25%`,
  - decay sharply to base LR by `~25-40%`,
  - after KD-stop trigger, apply additional fast drop to CE-finish LR floor.

> [!TIP]
> **KD Sweep Priors (Next Runs Must Use)**
> - `KD_STOP_MODE=val_margin`, `KD_STOP_MARGIN=0.05`, `KD_STOP_MIN_STEP=25`.
> - temperature search stays narrow (`1.0-2.0`).
> - include LR-focused grid centered on boosted early phase:
>   - `kd_lr_boost_mult in {2.0, 3.0, 4.0, 5.0}`
>   - `kd_lr_boost_hold_frac in {0.15, 0.20, 0.25}`
>   - `kd_lr_boost_drop_to_base_frac in {0.30, 0.35, 0.40}`
> - rank configs by `steps_to_target` first, then stability of val-loss curve.

> [!WARNING]
> **Do Not Regress**
> - Do not treat KD as constant-weight add-on with CE LR unchanged.
> - Do not keep boosted LR deep into CE-only tail after KD is effectively done.
> - Do not disable dynamic KD normalization.

## Canonical Artifact Roots
- Local canonical root: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/final_wrapup_20260214T024955Z`
- Shared-disk mirror root: `/data/nanogpt_wrapup_20260214T024955Z`
- Remote campaign root (latest): `/root/projects/nanogpt-speedrun/runs/codex_worker/campaigns/campaign_25m127m_20260213T2241Z`

## Runtime Anchors
- CE student baseline (H100, full 1555):
  - log: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/remote_h100/server_logs/aadab377-e981-419d-907b-470c4e770f33.txt`
  - final val: `3.2783`
  - train time: `694015 ms` (`446.31 ms/step avg`)
- Teacher budget from 10% rule:
  - `69402 ms` total training time
  - equivalent full-1555 average budget: `44.63 ms/step`
  - practical implication: compact full runs or moderate-step wider teachers are now viable

## Active Targets
1. Measure CE baseline timing with same runtime stack used for new experiments (sanity rerun if needed).
2. Build teacher candidate sweep optimized for low runtime cost first.
3. Keep only teachers that satisfy:
   - `projected_teacher_time_ms <= 69402`
4. From survivors, select teacher with best validation trajectory.
5. Use selected teacher for KD sweeps and full KD runs.
6. Final success metric:
   - `steps_saved = ce_steps_to_target - kd_steps_to_target`
   - with teacher cost budget still satisfied.
7. Current best budget-compliant teacher candidate:
   - `MODEL_NUM_LAYERS=3`, `MODEL_NUM_HEADS=6`, `MODEL_HEAD_DIM=128`, `MODEL_DIM=768`, `BIGRAM_VOCAB_SIZE=2`
   - checkpointed full run: best/final val `4.0690` at step `250` (train_time `56620 ms`)
   - teacher checkpoint dir: `/data/campaign_budget_teacher_10pct/teacher_budget10_ckpt_L3_H6_D128_250_20260214T212840Z`
   - checkpointed run exists with `state_best.pt` and `state_step000175.pt` (I/O increases logged train_time)

## Performance Analysis: Step-Wise Wins vs Wall-Time Regression

This is a documentation-only analytical addendum focused on the current KD paradox:
- KD can look strong on loss-per-step in selected runs.
- KD is currently slower in wall-time than CE in recent runs.
- Most likely reason is not only kernels; the "small teacher" is still large in parameter count and compute.

### Evidence Sources (Locked)
- CE reference (target-4.07 campaign):
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/v3/ce_ref_live.log`
- Current KD runs:
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/v3/V01_live.log`
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/fast12/F01_live.log`
- Historical step-strong KD run:
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/remote_worker/kd_s1a_20260213T2303Z/campaign/kd_sweep_stage1/subset_a/kd_s1a_20260213T2303Z_kd_s1a_4.log`
- Chart references:
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/all_sweeps_val_steps.png`
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/all_sweeps_val_walltime.png`
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/all_sweeps_trainhard_norm_steps.png`
  - `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/hillclimb_sweeps/all_sweeps_trainhard_norm_walltime.png`

### Locked Metrics

| run | status | steps_to_4.07 | time_to_4.07 | median_step_ms (steady) | p90_step_ms | latest_val_point |
|---|---|---:|---:|---:|---:|---|
| CE reference | complete | 650 | 212.851s | 251.0ms | 462.0ms | `step 650, val 4.0300` |
| KD V01 | provisional/incomplete | not reached | not reached | 768.0ms | 769.0ms | `step 275, val 4.5282, time 224.465s` |
| KD F01 | provisional/incomplete | not reached | not reached | 768.0ms | 1746.0ms | `step 625, val 4.2231, time 605.279s` |

Historical step-wise strong KD reference:
- KD run (`kd_s1a_4`) at step `155`: `val 4.0861`, `train_time 442.054s`.
- CE comparison around same step (from CE reference): step `150` gives `val 4.8467`, `train_time 51.207s`.
- Conclusion: **better loss at same step, much worse wall-time**.

### What Is Good
- Step-wise signal can improve much faster in selected KD settings.
- Historical stage-1 run is evidence of strong step efficiency.

### What Is Failing
- Current wall-time efficiency regressed strongly versus CE.
- Recent KD runs did not hit `4.07` before CE-equivalent wall-time.

### Likely Root Causes (Ranked)
- `R1 (Primary)`: Teacher is not truly small in compute.
  - Logged params in current setup: student `350.653M`, teacher `296.389M`.
  - Reducing depth alone did not reduce dominant parameter/compute components enough.
- `R2`: KD adds teacher forward + soft-loss compute every train step.
- `R3`: Kernel-path instability risk under KD path (seen by higher p90 spikes in F01).
- `R4`: Schedule/stop interactions can hurt wall-time if KD remains active too long without enough wall-time payoff.

### Immediate Fix Direction
- Constrain teacher to genuinely low-compute regime first (true compute budget, not depth-only).
- Keep reporting both metrics together: `steps_to_target` and `time_to_target`.
- Run short profiling isolates before full sweeps:
  - CE only
  - CE + teacher inference only
  - CE + KD soft loss
- Keep `provisional/incomplete` labeling on live runs until completion.

### Important Changes to Public APIs/Interfaces/Types
- No trainer API changes in this pass.
- No script interface changes in this pass.
- Documentation-only update in `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/HILLCLIMB_GOALS_TARGETS.md`.

### Validation and Review Checklist
- All numeric claims in this section match the cited logs.
- V01/F01 are clearly labeled as provisional/incomplete.
- Section includes both step-based and wall-time-based interpretation.
- Chart file paths referenced in this section exist.
- Existing goals/targets are unchanged and not contradicted.

### Assumptions and Defaults
- This is a quick analytical addendum, not a deep profiling report.
- Incomplete runs are included and explicitly marked provisional.
- The target document for this analysis is `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/HILLCLIMB_GOALS_TARGETS.md`.
- Root-cause discussion is evidence-based hypothesis, not final proof.

### Kernel R&D Update (2026-02-15)
- Goal: remove KD kernel-path regression and re-measure CE vs KD wall-time on H100.
- Runtime controls added in trainer:
  - `REQUIRE_FLASH_ATTN` (fail fast if flash backend unavailable)
  - `KD_COMPILE_TEACHER` (teacher compile path)
  - `KD_APPLY_EVERY_STEPS`, `KD_MICROBATCHES_PER_STEP` (sparse KD compute)
- Validation that flash path was active in tested KD runs:
  - `FLASH_ATTN_BACKEND=fa3_kernels HAS_FLASH_ATTN=True`
  - `teacher_force_sdpa=False compile_teacher=True`

| run | setting | final step | final val | final train time | core median step-delta |
|---|---|---:|---:|---:|---:|
| short CE | baseline | 60 | 4.9544 | 60.755s | 672ms |
| short KD dense | KD every step, full microbatches | 60 | 4.9776 | 186.184s | 2959ms |
| short KD sparse | `every=2`, `micro=1` | 60 | 4.9886 | 92.987s | 682ms |
| long CE | baseline | 155 | 4.1248 | 107.242s | 465ms |
| long KD sparse | `every=2`, `micro=1` | 155 | 4.1318 | 142.344s | 618ms |
| long KD sparse + LR boost | sparse + `KD_LR_BOOST_MULT=2.0` early | 155 | 4.2047 | 142.829s | 619ms |

Conclusions:
- Dense KD regression was kernel/runtime-path dominated and is now avoided by sparse KD controls.
- Sparse KD brought per-step startup-window speed near CE (10-40 window medians ~252-254ms) but did not beat CE wall-time to the same quality in current settings.
- Early LR boost in this exact sparse setup did not improve quality-time tradeoff.

Artifacts:
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_rd_20260214T232839Z/`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_rd_long_20260214T233710Z/`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_rd_summary_20260215.csv`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_long_compare_20260215.csv`

### KD Soft-Loss Top-K Update (2026-02-15, follow-up)
- Added optional KD soft-loss controls:
  - `KD_SOFT_LOSS_TOPK` (teacher-top-k distillation support)
  - `KD_SOFT_LOSS_TOKEN_STRIDE` (token subsampling in KD soft-loss)
- New long-run comparisons (155 steps):
  - `KD_topk256_s1`: `val@155=4.1336`, `time@155=142.598s`
  - `KD_topk256_s2`: `val@155=4.1295`, `time@155=135.373s`
- Best current KD wall-time/quality variant:
  - `KD_topk256_s2` (`KD_SOFT_LOSS_TOPK=256`, `KD_SOFT_LOSS_TOKEN_STRIDE=2`, sparse KD active every 2 steps on 1 microbatch)
- Current status vs CE baseline at 155 steps:
  - CE: `val@155=4.1248`, `107.242s`
  - Best KD: `val@155=4.1295`, `135.373s`
  - Quality is very close; wall-time still behind CE, but gap reduced vs dense KD.

Additional artifacts:
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_long_compare_v2_20260215.csv`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_long_val_steps_v2.png`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_long_val_walltime_v2.png`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_long_step_delta_v2.png`

### KD Compile-Spike Fix (2026-02-15, warmup parity patch)
- Root cause identified:
  - warmup used dense KD logic even when runtime used sparse KD (`KD_APPLY_EVERY_STEPS`, `KD_MICROBATCHES_PER_STEP`),
  - this missed compile variants and caused huge runtime recompiles at schedule transitions (notably steps ~28 and ~55).
- Fix applied in trainer:
  - warmup now mirrors runtime KD branch selection exactly (step-level KD active/inactive + microbatch-level KD gating).
- Outcome on H100 (compile/warmup ON, 80 steps, same config):

| run | final train_time | final val | step-28 delta | step-55 delta | time ratio vs CE |
|---|---:|---:|---:|---:|---:|
| CE baseline | 37.826s | 4.6666 | 470ms | 684ms | 1.00x |
| KD full (pre-patch) | 64.817s | 4.6684 | 8370ms | 9410ms | 1.71x |
| KD full (post-patch) | 40.385s | 4.6559 | 469ms | 808ms | 1.07x |
| Teacher infer-only (post-patch) | 37.604s | 4.6761 | 475ms | 693ms | 0.99x |

Interpretation:
- Teacher inference itself is now effectively CE-speed (infer-only ~= CE).
- The previous KD wall-time blowup was primarily compile-variant mismatch at transitions.
- After fix, full KD overhead is close to CE for this setting; remaining overhead is mostly soft-loss compute.

Artifacts:
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_fix_20260215/kd_warmup_patch_summary.csv`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_fix_20260215/kd_warmup_patch_step_delta.png`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_fix_20260215/rd5_ce_warm_20260215T015359Z.txt`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_fix_20260215/rd5_fullkd_warm_20260215T015359Z.txt`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_fix_20260215/rd6_fullkd_warm_patch_20260215T015937Z.txt`
- `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_kernel_fix_20260215/rd6_inferonly_warm_patch_20260215T020241Z.txt`

## KD Soft-Loss Debug Snapshot (2026-02-15)

Locked findings from the latest exact-KL memory/timing probes:

- Unchunked exact KL probe (`KD_SOFT_LOSS_TOKEN_CHUNK=0`):
  - local artifacts: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_unchunked_probe/20260215T143633Z`
  - result: OOM during warmup soft-loss path before runtime `step:` logs.
  - error: tried `9.21 GiB`, free `3.32 GiB`, shortfall `~5.89 GiB`.
  - implication: no stable runtime step-ms available from this run.
- Chunk=8192 exact KL probe:
  - local artifacts: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_chunk_probe/20260215T144819Z`
  - result: OOM during warmup.
  - error: tried `1.54 GiB`, free `~0.30 GiB` (`298.12 MiB`).
- Chunk=1024 exact KL probe (random teacher model with same teacher architecture):
  - local artifacts: `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/kd_chunk_probe/20260215T145059Z`
  - status: no OOM up to step `102`.
  - median step delta (drop first 20): `~3359 ms`, p90 `~6624 ms`.
  - median KD profile components:
    - student: `~147.6 ms`
    - teacher: `~61.8 ms`
    - soft: `~454.0 ms`
    - backward: `~2673.2 ms` (dominant)
  - median peak VRAM:
    - allocated: `~56.3 GiB`
    - reserved: `~78.3 GiB`

Interpretation (evidence-based, provisional):
- Current full-KD exact-path slowdown is dominated by backward under heavy memory pressure, not teacher inference alone.

## Run Ledger

| run_id | mode | teacher_params_m | teacher_steps | teacher_train_time_ms | alpha | temp | kd_lr_boost | student_steps | best_val | final_val | steps_to_target | status | artifact_dir |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| aadab377-e981-419d-907b-470c4e770f33 | ce_baseline | - | - | 694015 | - | - | - | 1555 | 3.2783 | 3.2783 | baseline_ref | done | `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/remote_h100/server_logs/aadab377-e981-419d-907b-470c4e770f33.txt` |
| teacher25m_ce_20260213T2241Z_teacher25m_ce | teacher_ce_archival | 163.448 | 1555 | 362472 | - | - | - | - | 3.8721 | 4.0500 | - | done (archival) | `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/remote_worker/teacher25m_ce_20260213T2241Z/campaign/teacher` |
| student3l_ce_ckpt_20260214T014707Z_student3l_ce | teacher_candidate_proxy | 489.555 | 1000 (partial) | 184101 | - | - | - | - | 3.9429 | partial | - | partial | `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/remote_worker/student3l_ce_ckpt_20260214T014707Z/campaign/student_3layer_ce` |
| teacher_budget_L12_H6_D8_155_20260214T203704Z | teacher_budget_test | ~14.83 | 155 | 32917 | - | - | - | - | 6.4735 | 6.4735 | - | done (budget-pass, weak quality) | `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_budget_runs/teacher_budget_L12_H6_D8_155_20260214T203704Z.txt` |
| teacher_budget_L3_H6_D128_250_20260214T204116Z | teacher_budget_test | 296.389 | 250 | 66542 | - | - | - | - | 4.0686 | 4.0686 | - | done (budget-pass, best current val) | `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_budget_runs/teacher_budget_L3_H6_D128_250_20260214T204116Z.txt` |
| teacher_budget10_ckpt_L3_H6_D128_250_20260214T212840Z | teacher_budget_ckpt_best | 296.389 | 250 | 56620 | - | - | - | - | 4.0690 | 4.0690 | - | done (budget-pass, checkpoint ready for KD) | `/data/campaign_budget_teacher_10pct/teacher_budget10_ckpt_L3_H6_D128_250_20260214T212840Z` |
| teacher_budget_L3_H6_D128_250_20260214T204116Z@slice177 | teacher_budget_slice | 296.389 | 177 | 34539 | - | - | - | - | 4.3736 | 4.3736 | - | under-budget slice (legacy 5% reference) | `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_budget_runs/teacher_budget_summary.csv` |
| teacher_budget_ckpt_L3_H6_D128_175_20260214T204613Z | teacher_budget_ckpt | 296.389 | 175 | 39767 | - | - | - | - | 4.2473 | 4.2473 | - | done (checkpoint available, budget-pass under 10%) | `/data/campaign_budget_teacher_5pct/teacher_budget_ckpt_L3_H6_D128_175_20260214T204613Z` |

## Promote/Reject Checklist (Teacher Candidates)
- Promote only if all are true:
  - runtime budget satisfied (`<=69402 ms`, measured or projected),
  - stable downward validation over first checkpoints,
  - no OOM fallback required for baseline config,
  - deterministic artifacts complete.
- Reject if any are true:
  - budget miss,
  - flat/divergent validation early,
  - missing artifacts.

## Stop Conditions
- Teacher search stop: at least one teacher candidate meets runtime budget and has acceptable val curve.
- KD campaign stop: `kd_steps_to_target` computed vs CE baseline or explicit miss reason.
- Final stop: report contains teacher budget compliance + CE vs KD `steps_to_target`.

## Immediate Next Actions
1. Use checkpoint `/data/campaign_budget_teacher_10pct/teacher_budget10_ckpt_L3_H6_D128_250_20260214T212840Z/state_best.pt` as KD teacher.
2. Implement KD-aware LR handoff in trainer: boosted LR while KD active, sharp drop once KD-stop condition is reached.
3. Launch LR-prioritized KD sweep (short horizon first), then promote best config to longer run.
4. Compare against CE-at-matched-steps and report early `steps_saved` trend.
5. Keep all teacher/KD candidates in `/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt/logs/teacher_budget_runs/teacher_budget10_summary.csv`.
