import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
from validation.utils import ImageFolder

################################################### For making the input to the Diffusion Model ###################################################
import os
import random
import shutil

def copy_random_images(source_path, output_path, mipc=300):
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)

    # 소스 경로에서 모든 클래스 폴더 가져오기
    classes = os.listdir(source_path)

    # 각 클래스에 대해 이미지 복사
    for class_name in classes:
        class_input_path = os.path.join(source_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # 클래스 폴더에 있는 이미지 수와 mipc 중 작은 값을 사용
        images = os.listdir(class_input_path)
        num_images_to_copy = min(mipc, len(images))  # 클래스 이미지 수와 mipc 중 작은 값 사용
        selected_images = random.sample(images, num_images_to_copy)  # 이미지 수보다 mipc가 크면 그만큼만 선택

        for img in selected_images:
            src = os.path.join(class_input_path, img)
            dst = os.path.join(class_output_path, img)
            shutil.copy2(src, dst)

    print(f"모든 이미지가 {output_path}에 성공적으로 복사되었습니다.")
    



def init_images(args, model=None):
    trainset = ImageFolder(
        classes=args.classes,
        ipc=args.mipc * 2,
        shuffle=True,
        root=args.train_dir,
        transform=None,
    )

    trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(300),  # 이미지 가운데에서 300x300 크기로 자르기
            # MultiRandomCrop(
            #     num_crop=args.num_crop, size=300, factor=args.factor
            # ),
            normalize,
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        # 각 클래스별로 손실 계산
        class_images = {class_id: [] for class_id in range(args.nclass)}  # args.classes 대신 args.nclass 사용
        for i in range(images.size(0)):
            output = model(images[i:i+1])
            loss = F.cross_entropy(output, labels[i:i+1], reduction='none')
            class_images[labels[i].item()].append((images[i], loss.item()))

        # 각 클래스별로 손실이 낮은 이미지 선택
        selected_images = []
        for class_id in range(args.nclass):
            class_images[class_id].sort(key=lambda x: x[1])
            selected_images += [img for img, loss in class_images[class_id][:args.mipc]]

        # 선택된 이미지를 텐서로 병합
        selected_images = torch.stack(selected_images)

        # 이미지 섞기 및 저장
        images = selected_images #mix_images(selected_images, 300, 1, int(args.mipc))
        save_images(args, denormalize(images), c)



def save_images(args, images, class_id):
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main(args):
    # print(args)
    # with torch.no_grad():
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    else:
        shutil.rmtree(args.syn_data_path)
        os.makedirs(args.syn_data_path)
    copy_random_images(args.train_dir, args.syn_data_path, args.mipc)

if __name__ == "__main__":
    pass