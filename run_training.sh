#!/bin/bash

# Default values for training parameters
DATAPATH=${DATAPATH:-"../data/2sigma/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5"}
ET=${ET:-5}
ETA=${ETA:-0}
REF=${REF:-"../ringerFPGA/Models_Scripts/Models_Scripts/references/mc21_13p6TeV.Run3_v1.40bins.ref.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/vqat_q7"}
TAG=${TAG:-"vqat_q7"}
BATCH_SIZE=${BATCH_SIZE:-1024}
SORTS=${SORTS:-10}
INITS=${INITS:-5}
SEED=${SEED:-512}

# Flag for dry-run
DRY_RUN_ARG=""
if [[ "$*" == *"--dry-run"* ]]; then
    DRY_RUN_ARG="--dry-run"
fi

echo "Starting training for $TAG..."
echo "Data: $DATAPATH"
echo "ET: $ET, ETA: $ETA"
echo "Output Directory: $OUTPUT_DIR"

# Run the CLI
python cli.py vqat_q7 run-training \
    --datapath "$DATAPATH" \
    --et "$ET" \
    --eta "$ETA" \
    --ref "$REF" \
    --output-dir "$OUTPUT_DIR" \
    --tag "$TAG" \
    --batch-size "$BATCH_SIZE" \
    --sorts "$SORTS" \
    --inits "$INITS" \
    --seed "$SEED" \
    $DRY_RUN_ARG

echo "Finished training for $TAG. Results saved in $OUTPUT_DIR/$TAG."
