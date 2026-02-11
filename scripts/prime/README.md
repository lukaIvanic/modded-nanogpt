# Prime Spot Workflow

This directory contains scripts for running `modded-nanogpt` on Prime spot instances with small-artifact persistence.

## Scripts

- `prime_bootstrap_remote.sh`
  - Installs `python3-pip`, `python3-venv`, creates `/root/projects/nanogpt-speedrun/.venv`, and installs training dependencies.
- `prime_sync_repo_and_data.sh`
  - Syncs local repo code to remote and uploads:
    - `data/fineweb10B/fineweb_train_000001.bin`
    - `data/fineweb10B/fineweb_val_000000.bin`
  - Verifies FineWeb shard headers (`magic=20240520`, `version=1`).
- `prime_launch_train.sh`
  - Creates one run directory at `/root/projects/nanogpt-speedrun/runs/<run_id>/`
  - Captures `command.sh`, `env.txt`, `train.log`, `gpu.csv`, `val_metrics.csv`, `status.json`
  - Launches `torchrun --standalone --nproc_per_node=2 train_gpt.py`.
- `prime_pull_small_artifacts.sh`
  - Pulls only small files from remote run directory into local `logs/prime_a100/<run_id>/`.
  - Optional `--watch` mode keeps syncing every 20 seconds.
- `prime_commit_push_on_eval.sh`
  - Watches local synced files and commits/pushes after new eval rows.

## Typical sequence

```bash
./scripts/prime/prime_bootstrap_remote.sh
./scripts/prime/prime_sync_repo_and_data.sh
RUN_ID=prime_smoke_$(date -u +%Y%m%dT%H%M%SZ) ./scripts/prime/prime_launch_train.sh
./scripts/prime/prime_pull_small_artifacts.sh "$RUN_ID" --watch
./scripts/prime/prime_commit_push_on_eval.sh "$RUN_ID"
```
