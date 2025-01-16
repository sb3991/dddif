#!/bin/bash

arch_names=("resnet18" "mobilenet_v2" "swin_t")
stud_names=("vit_b_16" "resnet18" "mobilenet_v2" "swin_t" "efficientnet_b0")

for arch_name in "${arch_names[@]}"; do
  for stud_name in "${stud_names[@]}"; do

    if [[ "$arch_name" == "resnet18" && "$stud_name" == "resnet18" ]]; then
      echo "Skip: ($arch_name, $stud_name)"
      continue
    fi
    
    if [[ "$arch_name" == "mobilenet_v2" ]]; then
      if [[ "$stud_name" == "mobilenet_v2" || "$stud_name" == "resnet18" || "$stud_name" == "efficientnet_b0" ]]; then
        echo "Skip: ($arch_name, $stud_name)"
        continue
      fi
    fi
    
    echo "Running: ($arch_name, $stud_name)"
    python ./main.py \
      --subset "imagenet-1k" \
      --arch-name "${arch_name}" \
      --factor 1 \
      --num-crop 5 \
      --mipc 120 \
      --ipc 50 \
      --stud-name "${stud_name}" \
      --re-epochs 100 \
      --pipeline "particle"

  done
done
