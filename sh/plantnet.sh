#!/bin/bash
# LTIC on Pl@ntNet-300K with a frozen plant-aware encoder.
# Once, before training:  python tools/make_plantnet_splits.py --root $DATA
# Other encoders:         BACKBONE=dinov2_l14 bash sh/plantnet.sh   (or clip_b32)

DATA=${DATA:-/path/to/plantnet_300K}   # the --root given to make_plantnet_splits.py
SPLITS=${SPLITS:-$DATA}                 # folder holding plantnet_{train,val,test}.txt
BACKBONE=${BACKBONE:-bioclip2}
EPOCHS=${EPOCHS:-200}
WORKERS=${WORKERS:-16}
BATCH=${BATCH:-256}
RUNS=${RUNS:-./data}                    # logs and checkpoints
CACHE_DIR=${CACHE_DIR:-./feat_cache}
CACHE_VIEWS=${CACHE_VIEWS:-5}

python CLIP_VIT_LONGTAIL.py \
  --arch clip_VIT \
  --backbone $BACKBONE \
  --mark plantnet_${BACKBONE}_bt${BATCH} \
  -dataset plantnetDataset \
  --data_path "$DATA" \
  --train_txt "$SPLITS/plantnet_train.txt" \
  --val_txt "$SPLITS/plantnet_val.txt" \
  --root_path "$RUNS" \
  -b $BATCH \
  --epochs $EPOCHS \
  --num_works $WORKERS \
  -p 100 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --beta 0.85 \
  --gamma 0.3 \
  --num_classes 1081 \
  --cache_features \
  --cache_dir "$CACHE_DIR" \
  --cache_views $CACHE_VIEWS
