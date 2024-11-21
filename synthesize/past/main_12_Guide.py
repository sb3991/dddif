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
                images = [os.path.join(cls_image_dir, img) for img in os.listdir(cls_image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
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


def classifier_guidance(pipe, classifier, latents, t, class_idx, guidance_scale):
    latents.requires_grad_(True)
    
    with torch.enable_grad():
        # latents를 이미지로 디코딩
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        
        # 이미지 전처리
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # 분류기 입력 크기에 맞게 조정 (224x224로 가정)
        # image_resized = transforms.Resize((32, 32))(image)
        image_resized = torch.nn.functional.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)

        # 분류기에 이미지 입력
        logits = classifier(image_resized)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected = log_probs[:, class_idx]
        
        # 그래디언트 계산
        grad = torch.autograd.grad(selected.sum(), latents)[0]
    
    # 그래디언트를 사용하여 latents 업데이트
    guidance = guidance_scale * grad
    latents = latents.detach() + guidance
    return latents

def generate_images(data_loader, pipe, classifier, output_dir, image_size, images_per_class, prompt_strength=7.5, classifier_guidance_scale=1.0, num_inference_steps=20):
    os.makedirs(output_dir, exist_ok=True)
    class_indices = {class_name: 0 for class_name in set([class_name for _, _, _, class_name in data_loader.dataset])}

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # num_inference_steps = 20
    print(f"추론 단계 수: {num_inference_steps}")
    print(f"Prompt 강도: {prompt_strength}, Classifier guidance 강도: {classifier_guidance_scale}")

    for batch_idx, (images, prompts, negative_prompts, class_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="이미지 생성 중"):
        pipe.scheduler.set_timesteps(num_inference_steps)
        
        images = images.to("cuda", dtype=torch.float16)
        latents = pipe.vae.encode(images).latent_dist.sample().to("cuda")
        latents = latents * pipe.vae.config.scaling_factor

        # Prompt 처리
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.tolist()
        text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to('cuda')
        
        # Negative prompt 처리
        if negative_prompts and isinstance(negative_prompts[0], str):
            uncond_inputs = tokenizer(negative_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_input_ids = uncond_inputs.input_ids.to('cuda')
        else:
            uncond_input_ids = tokenizer([""] * len(prompts), padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda')

        text_embeddings = pipe.text_encoder(text_input_ids)[0]
        uncond_embeddings = pipe.text_encoder(uncond_input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        noise = torch.randn_like(latents)
        latents = noise * pipe.scheduler.init_noise_sigma

        for step, t in enumerate(pipe.scheduler.timesteps):
            with torch.no_grad():
                latents_input = torch.cat([latents] * 2)
                noise_pred = pipe.unet(latents_input, t, encoder_hidden_states=text_embeddings)["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + prompt_strength * (noise_pred_text - noise_pred_uncond)
                latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

            # 분류기 가이던스 적용
            for i, class_name in enumerate(class_names):
                class_idx = int(class_name)
                latents[i:i+1] = classifier_guidance(pipe, classifier, latents[i:i+1], t, class_idx, classifier_guidance_scale)

        # 최종 latents를 디코딩하여 이미지 생성
        with torch.no_grad():
            generated_images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
            generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
            generated_images = generated_images.cpu().detach().permute(0, 2, 3, 1).numpy()

        for i, (img, class_name, prompt) in enumerate(zip(generated_images, class_names, prompts)):
            if class_indices[class_name] < images_per_class:
                img = Image.fromarray((img * 255).astype(np.uint8))
                img = transforms.Resize(image_size)(img)
                
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                img_index = class_indices[class_name]
                img.save(os.path.join(class_dir, f"Class_{class_name}_{img_index}.png"))
                
                # with open(os.path.join(class_dir, f"Class_{class_name}_{img_index}_prompt.txt"), "w") as f:
                #     f.write(f"Prompt: {prompt if isinstance(prompt, str) else str(prompt)}\n")
                #     f.write(f"Prompt Strength: {prompt_strength}\n")
                #     f.write(f"Classifier Guidance Scale: {classifier_guidance_scale}\n")

                class_indices[class_name] += 1

        if all(class_indices[class_name] >= images_per_class for class_name in class_indices):
            print("모든 클래스에 대해 충분한 이미지가 생성되었습니다. 프로세스를 종료합니다.")
            break

    print("이미지 생성 프로세스가 완료되었습니다.")

def main(args):
    image_root = args.syn_data_path
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
        prompt_root = '/default/path/to/labels.txt'
    
    additional_prompt_dir = os.path.join(image_root, 'prompts')  # 추가 프롬프트 디렉토리 경로
    additional_path = 'Diffusion_image'
    parent_dir = remove_last_directory(args.syn_data_path)
    output_dir = os.path.join(parent_dir, additional_path)
    batch_size = 4  # Reduce batch size to decrease memory usage
    images_per_class = args.mipc
    prompt_strength=args.prompt_strength
    classifier_scale=args.classifier_scale
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    num_inference_steps=args.num_inference_steps
    dataset = CustomImagePromptDataset(image_root, prompt_root, additional_prompt_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"데이터셋 크기: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어 있습니다. 이미지와 프롬프트 파일이 올바르게 로드되는지 확인하세요.")

    model_id = "runwayml/stable-diffusion-v1-5"
    max_width, max_height = get_max_image_size(image_root)
    image_size = (max(max_width, max_height), max(max_width, max_height))

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")

    classifier = load_model_2(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    )
    classifier = classifier.to("cuda").half()
    classifier.eval()

    generate_images(data_loader, pipe, classifier, output_dir, image_size, images_per_class, prompt_strength, classifier_scale, num_inference_steps)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_path", type=str, required=True, help="Path to the root directory of images")
    parser.add_argument("--prompt_root", type=str, required=True, help="Path to the label prompt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of samples to generate")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--prompt_strength", type=float, default=7.5, help="Classifier guidance scale")
    parser.add_argument("--use_manual_class", action="store_true", help="Use manual class selection")
    parser.add_argument("--manual_class_id", type=int, default=0, help="Manual class ID to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--arch_name", type=str, required=True, help="Classifier architecture name")
    parser.add_argument("--subset", type=str, required=True, help="Dataset subset")
    parser.add_argument("--classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--image_size", type=int, default=32, help="Size of generated images")

    args = parser.parse_args()
    main(args)