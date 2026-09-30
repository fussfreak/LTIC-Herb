#!/bin/bash
# Evaluate the best Pl@ntNet-300K checkpoint on the official test split.

DATA=${DATA:-/path/to/plantnet_300K}
BACKBONE=${BACKBONE:-bioclip2}

python CLIP_VIT_LONGTAIL.py \
  --arch clip_VIT \
  --backbone $BACKBONE \
  --mark plantnet_${BACKBONE}_bt256 \
  -dataset plantnetDataset \
  --data_path $DATA \
  --val_txt $DATA/plantnet_test.txt \
  --resume ./data/plantnetDataset/plantnet_${BACKBONE}_bt256/model_best.pth.tar \
  -b 256 \
  --num_works 16 \
  --beta 0.85 \
  --gamma 0.3 \
  --num_classes 1081 \
  --evaluate
