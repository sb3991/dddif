python ./main.py \
--subset "imagenet-1k" \
--arch-name "mobilenet_v2" \
--factor 1 \
--num-crop 5 \
--mipc 120 \
--ipc 50 \
--stud-name "resnet18" \
--re-epochs 100 \
--pipeline "particle"
# vit : vit_b_16
# swin : swin_t