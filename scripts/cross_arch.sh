#!/bin/bash

arch_names=("swin_v2_t")
stud_names=("vit_b_16" "resnet18" "mobilenet_v2" "swin_v2_t" "efficientnet_b0")

for arch_name in "${arch_names[@]}"; do
  for stud_name in "${stud_names[@]}"; do
    echo "Running: ($arch_name, $stud_name)"
    python ./main.py \
      --subset "imagenet-1k" \
      --arch-name "${arch_name}" \
      --factor 1 \
      --num-crop 5 \
      --mipc 120 \
      --ipc 50 \
      --stud-name "${stud_name}" \
      --re-epochs 1000 \
      --pipeline "particle"

  done
done
