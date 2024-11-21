from torch.cuda.amp import autocast
import gc
import torch.nn as nn
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, DDIMScheduler
from tqdm.auto import tqdm
import random
from transformers import AutoTokenizer
from synthesize.utils import load_model, load_model_2


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(1, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.linear1.weight.device)
        t = t.float().view(-1, 1).half()  # float16으로 변환
        t = self.linear1(t)
        t = torch.sin(t)
        t = self.linear2(t)
        return t
    
    def to(self, device):
        self.linear1 = self.linear1.to(device)
        self.linear2 = self.linear2.to(device)
        return super().to(device)


class LatentMLP(nn.Module):
    def __init__(self, input_channels, input_size, hidden_dim, num_classes, num_layers=3, activation='relu', dropout=0.5):
        super(LatentMLP, self).__init__()
        
        self.input_dim = input_channels * input_size * input_size
        self.time_embed = TimeEmbedding(hidden_dim)
        
        layers = []
        current_dim = self.input_dim + hidden_dim  # 시간 임베딩을 위해 차원 증가
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError("Unsupported activation function")
            
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        x = x.half()  # float16으로 변환
        x = x.view(x.size(0), -1)
        t_embed = self.time_embed(t)
        
        # t_embed를 x의 배치 크기에 맞게 확장
        if t_embed.size(0) == 1:
            t_embed = t_embed.expand(x.size(0), -1)
        
        x = torch.cat([x, t_embed], dim=1)
        return self.network(x)
    
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

# class ClassifierGuidedStableDiffusionPipeline(StableDiffusionPipeline):
#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
#         pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
#         return cls(**pipeline.components)

#     def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder=None, requires_safety_checker=True):
#         super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder, requires_safety_checker)
#         self.classifier = None

#     def set_classifier(self, classifier):
#         self.classifier = classifier


#     def step(self, model_output, timestep, sample, **kwargs):
#         # Classifier 그래디언트 계산 및 노이즈 예측 수정
#         if self.classifier is not None:
#             sample.requires_grad_()
#             class_scores = self.classifier(sample, timestep)
#             class_labels = kwargs.get('class_labels')
#             if class_labels is not None:
#                 class_loss = -torch.log_softmax(class_scores, dim=1).gather(1, class_labels.unsqueeze(1)).squeeze()
#                 grad = torch.autograd.grad(class_loss.sum(), sample)[0]
                
#                 gradient_scale = kwargs.get('gradient_scale', 1.0)
#                 print("gradient_scale", gradient_scale)
#                 model_output = model_output - gradient_scale * self.scheduler.alphas_cumprod[timestep].sqrt() * grad
#         else:
#             print("No Classifier!")

#         # 기존 scheduler step 호출
#         return self.scheduler.step(model_output, timestep, sample, **kwargs)

#     def _denoising_step(self, model_output, timestep, sample, **kwargs):
#         return self.step(model_output, timestep, sample, **kwargs)

#     @torch.no_grad()
#     def __call__(self, *args, **kwargs):
#         # class_labels와 gradient_scale을 kwargs에 추가합니다.
#         class_labels = kwargs.pop('class_labels', None)
#         gradient_scale = kwargs.pop('gradient_scale', 1.0)

#         # 부모 클래스의 __call__ 메서드 호출
#         result = super().__call__(*args, **kwargs)

#         # 결과 처리 및 반환
#         return result

class ClassifierGuidedStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(**pipeline.components)

    def set_classifier(self, classifier):
        self.classifier = classifier

    @torch.no_grad()
    def __call__(self, prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, negative_prompt=None, num_images_per_prompt=1, eta=0.0, generator=None, latents=None, class_labels=None, gradient_scale=1.0, **kwargs):        # 텍스트 임베딩 생성
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        # 초기 노이즈 생성
        latents = self.prepare_latents(4 * num_images_per_prompt, 4, height, width, generator, latents) #batch_size

        # 디노이징 루프
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            # 노이즈 예측
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample

            # Classifier guidance 적용
            if self.classifier is not None and class_labels is not None:
                latents.requires_grad_()
                class_scores = self.classifier(latents, t)
                class_loss = -torch.log_softmax(class_scores, dim=1).gather(1, class_labels.unsqueeze(1)).squeeze()
                grad = torch.autograd.grad(class_loss.sum(), latents)[0]
                noise_pred = noise_pred - gradient_scale * self.scheduler.alphas_cumprod[t].sqrt() * grad
                latents = latents.detach()

            # 스케줄러 스텝
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 이미지 디코딩
        images = self.decode_latents(latents)

        return images

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(**pipeline.components)    

def generate_images(data_loader, pipe, output_dir, image_size, images_per_class, classifier_scale, prompt_strength, num_inference_steps):
    os.makedirs(output_dir, exist_ok=True)
    class_indices = {class_name: 0 for class_name in set([class_name for _, _, _, class_name in data_loader.dataset])}
    class_to_idx = {name: idx for idx, name in enumerate(sorted(set(class_indices.keys())))}

    total_generated = 0
    total_required = sum(images_per_class for _ in class_indices)
    print("prompt_strength", prompt_strength, "classifier_scale", classifier_scale, "num_inference_steps", num_inference_steps)

    with tqdm(total=total_required, desc="Generating images") as pbar:
        while any(class_indices[class_name] < images_per_class for class_name in class_indices):
            for images, prompts, negative_prompts, class_names in data_loader:
                images = images.to("cuda")
                prompts = list(prompts)
                negative_prompts = ['cartoon, anime, painting'] * len(prompts)

                # 클래스 이름을 인덱스로 변환
                class_labels = torch.tensor([class_to_idx[name] for name in class_names]).to("cuda")

                with torch.no_grad():
                    generated_images = pipe(
                        prompt=prompts,
                        negative_prompt=negative_prompts,
                        height=512,
                        width=512,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=prompt_strength,
                        class_labels=class_labels,
                        gradient_scale=classifier_scale,
                        # prompt=prompts,
                        # negative_prompt=negative_prompts,
                        # image=images,
                        # strength=0.8,
                        # guidance_scale=prompt_strength,
                        # num_inference_steps=num_inference_steps,
                        # class_labels=class_labels,
                        # gradient_scale=classifier_scale,
                        progress_bar=False
                    ).images

                    resized_images = generated_images

                for i, (img, class_name) in enumerate(zip(resized_images, class_names)):
                    if class_indices[class_name] < images_per_class:
                        class_dir = os.path.join(output_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)

                        img_index = class_indices[class_name]
                        img.save(os.path.join(class_dir, f"Class_{class_name}_{img_index}.png"))
                        class_indices[class_name] += 1
                        total_generated += 1
                        pbar.update(1)

                    if all(class_indices[class_name] >= images_per_class for class_name in class_indices):
                        break

                if total_generated >= total_required:
                    break

            if total_generated >= total_required:
                break

    print("Image generation completed.")




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
    batch_size = 1  # Reduce batch size to decrease memory usage
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


    model_path = "/home/sb/link/DD_DIF/additional_trained_models/cifar100_latent_with_t_mlp.pth"
    input_channels = 4  # Latent 벡터의 채널 수
    input_size = 64  # Latent 벡터의 크기
    hidden_dim = 512
    num_classes = 100  # CIFAR-100 클래스 수
    classifier = LatentMLP(input_channels=input_channels, input_size=input_size, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=4)
    classifier.load_state_dict(torch.load(model_path))
    classifier = classifier.to("cuda").half()
    classifier.eval()

    # scheduler = scheduler.to("cuda")  # 명시적으로 CUDA로 이동

    # scheduler.classifier = classifier
    # scheduler = ClassifierGuidedDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # scheduler = ClassifierGuidedDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    # pipe.scheduler.set_gradient_scale(classifier_scale)  # gradient_scale 설정
    # pipe.scheduler.set_classifier(classifier)

    # Initialize the custom pipeline with from_pretrained
    pipe = ClassifierGuidedStableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    ).to("cuda")

    pipe.set_classifier(classifier)

    pipe.set_progress_bar_config(disable=True)
    

    generate_images(data_loader, pipe, output_dir, image_size, images_per_class, classifier_scale, prompt_strength, num_inference_steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_path", type=str, required=True, help="Path to the root directory of images")
    parser.add_argument("--prompt_root", type=str, required=True, help="Path to the label prompt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for data loader")
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