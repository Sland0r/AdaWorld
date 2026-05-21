#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=build_shards
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_spatial_shards_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

TARGET_KIND=${TARGET_KIND:-difference}           # difference | flow
DUMP_DIR=${DUMP_DIR:-1}                          # 1=latent_actions_dump, 2=latent_actions_dump_2
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1024}
SHARD_ROOT=${SHARD_ROOT:-/scratch-shared/FoMo-Atomic-Actions/sharded_targets}
CACHE_DIR=${CACHE_DIR:-/scratch-shared/FoMo-Atomic-Actions/_cache}
MAX_SAMPLES=${MAX_SAMPLES:-}
START_INDEX=${START_INDEX:-0}
OVERWRITE=${OVERWRITE:-0}
STAGE_TO_TMPDIR=${STAGE_TO_TMPDIR:-0}

TARGET_ROOT=${TARGET_ROOT:-}
if [ -z "$TARGET_ROOT" ]; then
    if [ "$TARGET_KIND" = "difference" ]; then
        TARGET_ROOT=/scratch-shared/FoMo-Atomic-Actions/difference_dump/random_actions_data
    elif [ "$TARGET_KIND" = "flow" ]; then
        TARGET_ROOT=/scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data
    else
        echo "Unsupported TARGET_KIND: $TARGET_KIND" >&2
        exit 1
    fi
fi

CMD=(
    python -u new_stuff/build_spatial_shards.py
    --target_kind "$TARGET_KIND"
    --target_root "$TARGET_ROOT"
    --cache_dir "$CACHE_DIR"
    --shard_root "$SHARD_ROOT"
    --samples_per_shard "$SAMPLES_PER_SHARD"
    --dump_dir "$DUMP_DIR"
    --start_index "$START_INDEX"
)

if [ -n "$MAX_SAMPLES" ]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

if [ "$OVERWRITE" = "1" ]; then
    CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

if [ "$STAGE_TO_TMPDIR" = "1" ]; then
    if [ -z "${TMPDIR:-}" ]; then
        echo "STAGE_TO_TMPDIR=1, but TMPDIR is not set." >&2
        exit 1
    fi

    SRC_SHARDS="${SHARD_ROOT%/}/${TARGET_KIND}/dump_dir_${DUMP_DIR}"
    DST_SHARDS="${TMPDIR%/}/sharded_targets/${TARGET_KIND}/dump_dir_${DUMP_DIR}"
    echo "Staging shards to local disk: $SRC_SHARDS -> $DST_SHARDS"
    mkdir -p "$DST_SHARDS"
    cp -a "${SRC_SHARDS}/." "$DST_SHARDS/"
    echo "Local shard path: $DST_SHARDS"
fi
