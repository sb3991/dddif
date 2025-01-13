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