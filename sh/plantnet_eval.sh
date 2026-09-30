#!/bin/bash
# Evaluate the best Pl@ntNet-300K checkpoint (from sh/plantnet.sh) on the official test split.
# Results are logged to $RUNS/plantnetDataset/plantnet_${BACKBONE}_bt${BATCH}_test/train.log

DATA=${DATA:-/path/to/plantnet_300K}
SPLITS=${SPLITS:-$DATA}
BACKBONE=${BACKBONE:-bioclip2}
WORKERS=${WORKERS:-16}
BATCH=${BATCH:-256}
RUNS=${RUNS:-./data}

python CLIP_VIT_LONGTAIL.py \
  --arch clip_VIT \
  --backbone $BACKBONE \
  --mark plantnet_${BACKBONE}_bt${BATCH}_test \
  -dataset plantnetDataset \
  --data_path "$DATA" \
  --train_txt "$SPLITS/plantnet_train.txt" \
  --val_txt "$SPLITS/plantnet_test.txt" \
  --root_path "$RUNS" \
  --resume "$RUNS/plantnetDataset/plantnet_${BACKBONE}_bt${BATCH}/model_best.pth.tar" \
  -b $BATCH \
  --num_works $WORKERS \
  -p 100 \
  --beta 0.85 \
  --gamma 0.3 \
  --num_classes 1081 \
  --evaluate
