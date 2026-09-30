#!/bin/bash
# LTIC on Pl@ntNet-300K with a frozen plant-aware encoder.
# Once, before training:  python tools/make_plantnet_splits.py --root $DATA
# Other encoders:         BACKBONE=dinov2_l14 bash sh/plantnet.sh   (or clip_b32)

DATA=${DATA:-/path/to/plantnet_300K}
BACKBONE=${BACKBONE:-bioclip2}

python CLIP_VIT_LONGTAIL.py \
  --arch clip_VIT \
  --backbone $BACKBONE \
  --mark plantnet_${BACKBONE}_bt256 \
  -dataset plantnetDataset \
  --data_path $DATA \
  -b 256 \
  --epochs 200 \
  --num_works 16 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --beta 0.85 \
  --gamma 0.3 \
  --num_classes 1081 \
  --cache_features \
  --cache_views 5
