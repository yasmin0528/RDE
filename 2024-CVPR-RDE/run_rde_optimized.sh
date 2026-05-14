#!/bin/bash
# =============================================================
# Optimized training script
# Improvements over baseline:
#   1. ID Classification Loss (cross-entropy on detached BGE features)
#   2. Adaptive Margin for TAL (clean: full margin, noisy: reduced)
#   3. Soft Label Consensus (continuous GMM prob. instead of hard 0/1)
# =============================================================

# === Path Config ===
root_dir=/root/autodl-tmp/data
# ===================

tau=0.015
margin=0.1
noisy_rate=0.0      # 0.2 0.5 08
select_ratio=0.3
loss=TAL
id_loss_weight=0.5  # ID loss weight (0 to disable)
DATASET_NAME=RSTPReid

# dataset options: CUHK-PEDES ICFG-PEDES RSTPReid

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy

# === Build loss string ===
LOSS_STR="${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}"
if [ "$id_loss_weight" != "0" ]; then
    LOSS_STR="${LOSS_STR}+id${id_loss_weight}"
fi

CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE_optimized \
    --img_aug \
    --txt_aug \
    --batch_size 64 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir /root/autodl-tmp/run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names "$LOSS_STR" \
    --num_epoch 60 \
    --sampler random