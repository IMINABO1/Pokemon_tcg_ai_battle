#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION       # e.g. p12345 -- find yours: `sacctmgr show assoc user=$USER`
#SBATCH --partition=gengpu              # Quest's general GPU partition; see `sinfo` for others
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=field-zero-bc
#SBATCH --output=logs/%x_%j.out

module purge
module load apptainer

mkdir -p logs

# --nv wires up the host's NVIDIA driver into the container.
# Run from the CLONED repo's field_zero/ dir: binding $PWD over the image's
# /workspace/field_zero makes the cloned code (and its data_lake/checkpoints
# subdirs) the ones used -- the copy baked into the image is ignored.
apptainer exec --nv \
  --bind "$PWD:/workspace/field_zero" \
  field-zero.sif \
  python3 model/bc_train.py --decisions data_lake/decisions.parquet --epochs 5
