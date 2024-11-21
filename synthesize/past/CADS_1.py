import torch
from diffusers import StableDiffusionImg2ImgPipeline

class CADSStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def corrupt_text_cads(self, prompt_embeds, tau1=0.4, tau2=0.7, noise_scale=0.15, mixing_factor=1, rescale=True):
        gamma = self.cads_linear_schedule(tau1, tau2)
        return self.add_noise(prompt_embeds, gamma, noise_scale, mixing_factor, rescale)

    def cads_linear_schedule(self, tau1, tau2):
        cur_t = self.scheduler.num_train_timesteps - self.scheduler.num_inference_steps
        total_steps = self.scheduler.num_train_timesteps
        t = 1.0 - max(min(cur_t / total_steps, 1.0), 0.0)
        if t <= tau1:
            return 1.0
        if t >= tau2:
            return 0.0
        return (tau2 - t) / (tau2 - tau1)

    def add_noise(self, embeddings, gamma, noise_scale, mixing_factor, rescale):
        y_mean, y_std = torch.mean(embeddings), torch.std(embeddings)
        noise = noise_scale * torch.randn_like(embeddings)
        y_noisy = gamma * embeddings + (1 - gamma) * noise
        if rescale:
            y_noisy = (y_noisy - y_noisy.mean()) / y_noisy.std() * y_std + y_mean
        return mixing_factor * y_noisy + (1 - mixing_factor) * embeddings

    @torch.no_grad()
    def __call__(self, prompt=None, image=None, strength=0.8, guidance_scale=7.5, **kwargs):
        # Encode input prompt and image
        prompt_embeds = self.text_encoder(prompt)
        latents = self.vae.encode(image)

        # Apply CADS corruption to prompt embeddings
        prompt_embeds = self.corrupt_text_cads(prompt_embeds)

        # Denoise and decode latents
        for i in range(self.scheduler.num_inference_steps):
            latents = self.unet(latents, prompt_embeds)
        output_image = self.vae.decode(latents)

        return output_image

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 필요한 기본 매개변수 설정
        kwargs.setdefault('torch_dtype', torch.float16)
        if 'safety_checker' not in kwargs:
            kwargs['safety_checker'] = None
        
        # 부모 클래스의 from_pretrained 메서드를 호출
        return super(CADSStableDiffusionImg2ImgPipeline, cls).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
