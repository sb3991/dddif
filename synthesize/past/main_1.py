import os
from transformers import T5EncoderModel, BitsAndBytesConfig
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionImg2ImgPipeline, DDIMScheduler, StableDiffusion3Pipeline
import tqdm
import numpy as np
# For Making the Image Using Diffusion


class CustomImagePromptDataset(Dataset):
    def __init__(self, image_root, label_prompt_file, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.classes = sorted(os.listdir(image_root))
        self.image_paths = []
        self.prompts = []
        self.negative_prompts = []
        self.class_names = []

        with open(label_prompt_file, 'r') as f:
            label_prompts = f.readlines()
        label_prompts = [prompt.strip() for prompt in label_prompts]

        for cls in self.classes:
            cls_image_dir = os.path.join(image_root, cls)
            class_idx = int(cls)
            prompt = label_prompts[class_idx] if class_idx < len(label_prompts) else ""
            negative_prompt = " ".join([label_prompts[i] for i in range(len(label_prompts)) if i != class_idx])
            if os.path.isdir(cls_image_dir):
                images = [os.path.join(cls_image_dir, img) for img in os.listdir(cls_image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
                for img in images:
                    self.image_paths.append(img)
                    self.prompts.append(prompt)
                    self.negative_prompts.append(negative_prompt)
                    self.class_names.append(cls)
                print(f"클래스 '{cls}': {len(images)}개의 이미지와 프롬프트 '{prompt}' 로드됨")
            else:
                print(f"클래스 '{cls}': 이미지 디렉토리가 존재하지 않음.")

        print(f"총 이미지 수: {len(self.image_paths)}")
        print(f"총 프롬프트 수: {len(self.prompts)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        negative_prompt = self.negative_prompts[idx]
        class_name = self.class_names[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, prompt, negative_prompt, class_name

def get_max_image_size(image_root):
    max_width, max_height = 0, 0
    for root, _, files in os.walk(image_root):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
    return max_width, max_height

def generate_images(data_loader, pipe, output_dir, image_size, images_per_class):
    os.makedirs(output_dir, exist_ok=True)
    class_indices = {class_name: 0 for class_name in set([class_name for _, _, _, class_name in data_loader.dataset])}

    while any(class_indices[class_name] < images_per_class for class_name in class_indices):
        for batch_idx, (images, prompts, negative_prompts, class_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating images"):
            images = images.to("cuda")
            prompts = list(prompts)
            negative_prompts = list(negative_prompts)
            negative_prompts = ['cartoon, anime, painting'] * len(prompts)  

            with torch.no_grad():
                generated_images = pipe(prompt=prompts, negative_prompt=negative_prompts, image=images, strength=0.8, guidance_scale=7.5, progress_bar=False).images #, height=256, width=256
                resized_images = generated_images#[img.resize(image_size, Image.LANCZOS) for img in generated_images] #generated_images

            for i, (img, class_name) in enumerate(zip(resized_images, class_names)):
                if class_indices[class_name] < images_per_class:
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    img_index = class_indices[class_name]
                    img.save(os.path.join(class_dir, f"Class_{class_name}_{img_index}.png"))
                    class_indices[class_name] += 1

                if all(class_indices[class_name] >= images_per_class for class_name in class_indices):
                    break

def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir






def main(args):
    image_root = args.syn_data_path
    # prompt_root = '/home/sb/link/DD_DIF/label-prompt/cifar100_labels.txt'
    if 'cifar100' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/cifar100_labels.txt'
        # prompt_root = '/home/sb/link/DD_DIF/label-prompt/BLANK.txt'
        print("Load cifar100_labels.txt")
    elif 'cifar10' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/CIFAR-10_labels.txt'
        print("Load CIFAR-10_labels.txt")
    elif 'tinyimagenet' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/tiny_imagenet_labels_cleaned.txt'
        print("Load tiny_imagenet_labels_cleaned.txt")
    elif 'imagenet-100' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/imagenet100_labels.txt'
        print("Load imagenet100_labels.txt")
    else:
        prompt_root = '/default/path/to/labels.txt'
    additional_path = 'Diffusion_image'
    parent_dir = remove_last_directory(args.syn_data_path)
    output_dir = os.path.join(parent_dir, additional_path)
    # output_dir = os.path.join(args.syn_data_path, additional_path)


    batch_size = 4
    images_per_class = args.mipc
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = CustomImagePromptDataset(image_root, prompt_root, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"데이터셋 크기: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어 있습니다. 이미지와 프롬프트 파일이 올바르게 로드되는지 확인하세요.")

    model_id = "runwayml/stable-diffusion-v1-5"
    max_width, max_height = get_max_image_size(image_root)
    image_size = (max(max_width, max_height), max(max_width, max_height))

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None,  torch_dtype=torch.float16).to("cuda") #scheduler=scheduler,

    # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    # pipe.scheduler = scheduler

    pipe.set_progress_bar_config(disable=True)

    generate_images(data_loader, pipe, output_dir, image_size, images_per_class)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True, help="Path to the root directory of images")
    parser.add_argument("--prompt_root", type=str, required=True, help="Path to the label prompt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    parser.add_argument("--images_per_class", type=int, default=100, required=True, help="Number of images to generate per class")

    args = parser.parse_args()
    main(args)
