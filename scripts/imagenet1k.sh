python ./main.py \
--subset "imagenet-1k" \
--arch-name "swin_v2_t" \
--factor 1 \
--num-crop 5 \
--mipc 120 \
--ipc 50 \
--stud-name "swin_v2_t" \
--re-epochs 100 \
--pipeline "particle"
# vit : vit_b_16
# swin : swin_t