import os
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline
import tqdm
import random
from transformers import AutoTokenizer
from synthesize.utils import load_model, load_model_2
from sklearn.cluster import KMeans

# Custom Dataset Class
class CustomImagePromptDataset(Dataset):
    def __init__(self, image_root, label_prompt_file, additional_prompt_dir, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.classes = sorted(os.listdir(image_root))
        self.image_paths = []
        self.prompts = []
        self.negative_prompts = []
        self.class_names = []
        
        additional_path = 'prompts'
        parent_dir_ = remove_last_directory(self.image_root)
        additional_prompt_dir = os.path.join(parent_dir_, additional_path)
        
        # Load main label prompts
        with open(label_prompt_file, 'r') as f:
            label_prompts = f.readlines()
        label_prompts = [prompt.strip() for prompt in label_prompts]

        # Load additional prompts
        additional_prompts = {}
        for cls in self.classes:
            additional_prompt_file = os.path.join(additional_prompt_dir, f"{cls.zfill(5)}_captions.txt")
            if os.path.exists(additional_prompt_file):
                with open(additional_prompt_file, 'r') as f:
                    additional_prompts[cls] = f.readlines()
                additional_prompts[cls] = [prompt.strip() for prompt in additional_prompts[cls] if prompt.strip()]
            else:
                print(f"Warning: Additional prompt file not found for class {cls}")
                additional_prompts[cls] = []

        for cls in self.classes:
            cls_image_dir = os.path.join(image_root, cls)
            class_idx = int(cls)
            main_prompt = label_prompts[class_idx] if class_idx < len(label_prompts) else ""
            negative_prompt = " ".join([label_prompts[i] for i in range(len(label_prompts)) if i != class_idx])
            
            if os.path.isdir(cls_image_dir):
                images = [os.path.join(cls_image_dir, img) for img in os.listdir(cls_image_dir) if img.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]
                for img in images:
                    additional_prompt = random.choice(additional_prompts[cls]) if additional_prompts[cls] else ""
                    combined_prompt = f"{main_prompt} {additional_prompt}".strip()
                    self.image_paths.append(img)
                    self.prompts.append(combined_prompt)
                    self.negative_prompts.append(negative_prompt)
                    self.class_names.append(cls)

                print(f"클래스 '{cls}': {len(images)}개의 이미지 로드됨")
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
    
def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir

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


def save_latents(pipe, data_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Computing and saving latents")

    generator = torch.Generator(device="cuda").manual_seed(0)  # 일관성을 위해 고정된 시드 사용

    for images, _, _, class_names in data_loader:
        images = images.to("cuda", dtype=torch.float16)
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample(generator=generator).to("cuda")
            latents = latents * pipe.vae.config.scaling_factor
        
        for latent, class_name in zip(latents, class_names):
            class_name = class_name.item() if isinstance(class_name, torch.Tensor) else class_name
            class_dir = os.path.join(output_dir, str(class_name).zfill(5))  # 클래스 이름을 5자리 숫자로 패딩
            os.makedirs(class_dir, exist_ok=True)

            latent_index = len(os.listdir(class_dir))
            file_path = os.path.join(class_dir, f"latent_{latent_index}.pt")
            torch.save(latent.cpu(), file_path)
    
    print(f"Latent vectors saved in: {output_dir}")


def main(args):
    image_root = "/home/a5t11/ICML_Symbolic/Repo/Data/cifar10/train" #args.syn_data_path
    # image_root = "/home/a5t11/ICML_Symbolic/Repo/Data/cifar100/train/" #args.syn_data_path
    if 'cifar100' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/cifar100_labels.txt'
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
        prompt_root = '/home/sb/link/DD_DIF/label-prompt/cifar100_labels.txt'

    batch_size = 4  # Reduce batch size to decrease memory usage

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    num_inference_steps=args.num_inference_steps
    print(image_root)
    dataset = CustomImagePromptDataset(image_root, prompt_root, "/home/sb/link/DD_DIF/label-prompt/imagenet100_labels.txt", transform=transform)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"데이터셋 크기: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어 있습니다. 이미지와 프롬프트 파일이 올바르게 로드되는지 확인하세요.")

    model_id = "runwayml/stable-diffusion-v1-5"
    max_width, max_height = get_max_image_size(image_root)
    image_size = (max(max_width, max_height), max(max_width, max_height))

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")

    # Compute centroids
    # centroids = compute_latent_centroids(pipe, data_loader, num_clusters=1)
    save_latents(pipe, data_loader, '/home/sb/link/DD_DIF/saved_latetns/cifar10')

    # Save centroids to a directory
    # centroid_output_dir = "./centroids"
    # save_centroids(centroids, centroid_output_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()