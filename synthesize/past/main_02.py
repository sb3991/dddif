import os
import argparse
from transformers import OFATokenizer, OFAModel
from PIL import Image
import torch
import random


def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir


def main(args):
    # SimVLM 모델과 프로세서 로드
    model_name = "OFA-Sys/OFA-base"
    tokenizer = OFATokenizer.from_pretrained(model_name)
    model = OFAModel.from_pretrained(model_name)
    model.eval()

    additional_path = 'prompts'
    parent_dir = remove_last_directory(args.syn_data_path)
    output_dir = os.path.join(parent_dir, additional_path)

    if 'cifar100' in args.syn_data_path:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/cifar100_labels.txt'
        print("Load cifar100_labels.txt")
    elif 'cifar10' in args.syn_data_path:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/CIFAR-10_labels.txt'
        print("Load CIFAR-10_labels.txt")
    elif 'tinyimagenet' in args.syn_data_path:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/tiny_imagenet_labels_cleaned.txt'
        print("Load tiny_imagenet_labels_cleaned.txt")
    elif 'imagenet-100' in args.syn_data_path:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/imagenet100_labels.txt'
        print("Load imagenet100_labels.txt")
    elif 'imagenet-woof' in args.syn_data_path:
        prompt_root = '/home/a5t11/ICML_Symbolic/Repo/Data/label-prompt/Imagewoof.txt'
        print("Load imagenet100_labels.txt")
    else:
        prompt_root = '/default/path/to/labels.txt'

    # 결과 저장 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 클래스 이름 로드
    with open(prompt_root, 'r') as f:
        class_names = [line.strip().split(',') for line in f.readlines()]

    # 클래스 폴더 목록 가져오기
    class_folders = [f for f in os.listdir(args.syn_data_path) if os.path.isdir(os.path.join(args.syn_data_path, f))]

    for class_folder in class_folders:
        class_path = os.path.join(args.syn_data_path, class_folder)
        output_file = os.path.join(output_dir, f"{class_folder}_captions.txt")
        class_index = int(class_folder)  # Assuming class_folder is of the form '00000', '00001', etc.
        class_keywords = [keyword.strip() for keyword in class_names[class_index]]  # Remove leading/trailing spaces

        with open(output_file, 'w') as f_out:
            # 각 이미지 처리
            for image_name in os.listdir(class_path):
                if image_name.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(class_path, image_name)
                    image = Image.open(image_path).convert("RGB")

                    # 이미지 전처리 및 캡션 생성
                    inputs = tokenizer(image, return_tensors="pt").input_ids
                    outputs = model.generate(inputs)
                    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # # 30% 확률로 "at center" 추가
                    # if random.random() < 0.999:
                    #     caption += " from CIFAR 100 dataset"
                    
                    # 클래스 키워드 중 하나라도 캡션에 포함되어 있는지 확인
                    if any(keyword in caption for keyword in class_keywords):
                        # 캡션 출력 및 파일에 저장
                        print(f"{image_name}: {caption}")
                        f_out.write(f"{caption}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate captions for images in class folders.')
    parser.add_argument('--syn_data_path', type=str, required=True, help='Path to the synthetic data.')
    
    args = parser.parse_args()
    main(args)
