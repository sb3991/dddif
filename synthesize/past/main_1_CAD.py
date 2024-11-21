import os
from transformers import T5EncoderModel, BitsAndBytesConfig
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionImg2ImgPipeline, DDIMScheduler, StableDiffusion3Pipeline, DDPMScheduler, LMSDiscreteScheduler
from synthesize.CADS_1 import CADSStableDiffusionImg2ImgPipeline
import tqdm
import numpy as np
from synthesize.utils2 import TrainingFreeAttnProcessor
# For Making the Image Using Diffusion
class CustomImagePromptDataset(Dataset):
    def __init__(self, image_root, label_prompt_file, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.classes = sorted(os.listdir(image_root))
        self.image_paths = []
        self.prompts = []
        self.negative_prompts = ['cartoon, anime, painting']
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
            # negative_prompts = ['cartoon, anime, painting'] * len(prompts)  

            with torch.no_grad():
                generated_images = pipe(prompt=prompts, negative_prompt=negative_prompts, image=images, strength=0.8, guidance_scale=7.5, progress_bar=False).images #, height=256, width=256
                resized_images = generated_images #[img.resize(image_size, Image.LANCZOS) for img in generated_images] #generated_images

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


class BasicDDPMScheduler(DDPMScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, noise_pred, t, latents, **kwargs):
        """노이즈 예측을 기반으로 디노이징을 수행합니다."""
        # 기본 DDPM 스케줄러의 `step` 메소드 호출
        return super().step(noise_pred, t, latents, **kwargs)
    
class CADSLMSDiscreteScheduler(LMSDiscreteScheduler):
    def __init__(self, tau1=0.8, tau2=0.99, noise_scale=0.15, psi=0.9, rescale=True, **kwargs):
        super().__init__(**kwargs)
        self.tau1 = tau1
        self.tau2 = tau2
        self.noise_scale = noise_scale
        self.psi = psi
        self.rescale = rescale

    def cads_linear_schedule(self, t):
        """CADS annealing schedule function to calculate gamma."""
        if t.ndim == 0:
            t_1 = t.item()     
        else:
            t_1 =  t[0].item() 
        # print("t_1", t_1)    
        if t_1 <= self.tau1:
            return 1.0
        elif t_1 >= self.tau2:
            return 0.0
        else:
            return (self.tau2 - t_1) / (self.tau2 - self.tau1)

    def add_noise_CADS(self, latents, noise, t):
        """Add noise to embeddings based on the CADS strategy and current timestep."""
        gamma = self.cads_linear_schedule(t / 800)
        # print("gamma", gamma)
        y_noisy = gamma * latents + (1 - gamma) * noise
        if self.rescale:
            y_mean, y_std = torch.mean(latents), torch.std(latents)
            y_noisy = (y_noisy - y_noisy.mean()) / y_noisy.std() * y_std + y_mean
        return self.psi * y_noisy + (1 - self.psi) * latents

    def step(self, noise_pred, t, latents, **kwargs):
        """Perform a denoising step using CADS modifications."""
        noise = self.noise_scale * torch.randn_like(latents)
        modified_latents = self.add_noise_CADS(latents, noise, t)
        return super().step(noise_pred, t, modified_latents, **kwargs)

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
    # model_id = "stabilityai/stable-diffusion-2-1"
    # model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")




    # pipe = RescaledStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    # text_encoder = T5EncoderModel.from_pretrained(
    #     model_id,
    #     subfolder="text_encoder_3",
    #     quantization_config=quantization_config,
    # )


    max_width, max_height = get_max_image_size(image_root)
    image_size = (max(max_width, max_height), max(max_width, max_height))
    # pipe = StableDiffusion3Pipeline.from_pretrained(model_id,safety_checker=None, scheduler=scheduler, text_encoder_3=text_encoder,
    #     device_map="balanced", torch_dtype=torch.float16).to("cuda")


    # pipe = CADSStableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None,  torch_dtype=torch.float16).to("cuda") #scheduler=scheduler,
    # pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,  torch_dtype=torch.float16).to("cuda") #scheduler=scheduler,
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None,  torch_dtype=torch.float16).to("cuda") #scheduler=scheduler,

    cads_scheduler = CADSLMSDiscreteScheduler(
        num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
        beta_start=pipe.scheduler.config.beta_start,
        beta_end=pipe.scheduler.config.beta_end,
        tau1=0.9,
        tau2=1,
        noise_scale=0.15,
        psi=0.8,
        rescale=True
    )
    # BDS= BasicDDPMScheduler()
    pipe.scheduler = cads_scheduler

    # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    # pipe.scheduler = cads_scheduler

    # pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)


    # unet = pipe.unet
    # unet: UNet2DConditionModel
    # list_layers = []
    # for name, module in unet.named_modules():
    #     if 'attn' in name and all(x not in name for x in ['to_q', 'to_k', 'to_v', 'to_out']):
    #         list_layers.append(name)
    # processors = {name + '.processor': TrainingFreeAttnProcessor(unet, name) for name in list_layers}
    # unet.set_attn_processor(processors)

    # unet.set_attn_processor(
    #     {name: TrainingFreeAttnProcessor(name, unet) for name in list_layers}
    # )
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
