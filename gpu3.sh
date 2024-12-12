CUDA_VISIBLE_DEVICES=3

RUN="sh"
FILE="/home/sb/link/DD_DIF/scripts/tinyimagenet_10ipc_resnet-18-modified_to_resnet-18-modified_particle_cr5.sh"
#FILE="/home/sb/link/DD_DIF/scripts/imagenet-100_10ipc_conv6_to_conv6_cr5.sh"
#FILE="/home/sb/link/DD_DIF/scripts/imagenet-1k_10ipc_resnet-18_to_resnet-50_cr1.sh"
$RUN $FILE


#for file in /home/sb/link/DD_DIF/scripts/*.sh; do
#    bash "$file"
#done