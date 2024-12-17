CUDA_VISIBLE_DEVICES=2

RUN="sh"
#FILE="/home/sb/link/DD_DIF/scripts/cifar100_1ipc_conv3_to_conv3_cr5.sh"
#FILE="/home/sb/link/DD_DIF/scripts/imagenet-100_10ipc_conv6_to_conv6_cr5.sh"
FILE="/home/sb/link/DD_DIF/scripts/tinyimagenet_10ipc_resnet-18-modified_to_resnet-18-modified_particle_cr5.sh"
$RUN $FILE
