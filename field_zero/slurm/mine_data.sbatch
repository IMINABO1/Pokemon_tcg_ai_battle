#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION       # e.g. p12345 -- find yours: `sacctmgr show assoc user=$USER`
#SBATCH --partition=short               # CPU-only work: fetching/parsing JSON, no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=field-zero-mine
#SBATCH --output=logs/%x_%j.out

module purge
module load apptainer

mkdir -p logs data_lake checkpoints

# Kaggle needs credentials to list/download the episode datasets. Put your OWN
# kaggle.json at ~/.kaggle/kaggle.json on Quest before running this (never
# share someone else's token) -- bind-mount it read-only into the container.
# Run this from the CLONED repo's field_zero/ dir. Binding $PWD over the
# image's own /workspace/field_zero makes the cloned (git-pullable) code the
# one that runs -- the copy baked into the image at build time is ignored.
# EN_Card_Data.csv lives at the repo root, one level up, so it needs its own
# bind (it was never in the Docker build context).
apptainer exec \
  --bind "$PWD:/workspace/field_zero" \
  --bind "$PWD/../EN_Card_Data.csv:/workspace/EN_Card_Data.csv:ro" \
  --bind "$HOME/.kaggle:/root/.kaggle:ro" \
  field-zero.sif \
  bash -c "
    set -e
    python3 mining/fetch_episodes.py --since 2026-06-01 --out data_lake/raw
    python3 mining/parse_replay.py --raw data_lake/raw --out data_lake/parsed
    python3 mining/build_dataset.py --parsed data_lake/parsed --out data_lake --carddb /workspace/EN_Card_Data.csv
    python3 mining/audit_leakage.py --decisions data_lake/decisions.parquet
  "
