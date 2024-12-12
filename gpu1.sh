CUDA_VISIBLE_DEVICES=1

RUN="sh"
#FILE="/home/sb/link/DD_DIF/scripts/cifar100_1ipc_conv3_to_conv3_cr5.sh"
#FILE="/home/sb/link/DD_DIF/scripts/imagenet-100_10ipc_conv6_to_conv6_cr5.sh"
FILE="/home/sb/link/DD_DIF/scripts/imagenet-1k_10ipc_resnet-18_to_resnet-18_cr5.sh"
$RUN $FILE


#for file in /home/sb/link/DD_DIF/scripts/*.sh; do
#    bash "$file"
#done