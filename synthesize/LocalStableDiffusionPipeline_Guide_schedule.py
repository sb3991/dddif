import PIL
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
# from diffusers.utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import StableDiffusionPipeline

class LocalStableDiffusionPipeline(StableDiffusionImg2ImgPipeline): # 

    total_accuracy = 0
    accuracy_count = 0
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        image=None,
        strength=0.8,
        num_inference_steps=50,
        guidance_scale=7.5,
        classifier_guidance_scale=1.0,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        output_type="pil",
        class_labels=None,
        gradient_scale=1.0,
        return_dict=True,
        callback=None,
        callback_steps=1,
        early_stage_end=0.9,
        late_stage_start=0.4,
        **kwargs,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 4 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        classifier_guidance_scale=gradient_scale

        if class_labels is not None:
            class_labels = torch.tensor([int(label) for label in class_labels], device=device)
            # print("Load Labels", class_labels)
        else:
            class_labels = None


        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps ### FoR OG
        # print("num_inference_steps", num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device) ### FoR Im2Im
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
        )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        max_timestep = timesteps[0].item()  # timesteps는 일반적으로 내림차순으로 정렬되어 있습니다.
        # print("max_timestep", max_timestep)


        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # early_stage_end=0.9
            # late_stage_start=0.4
            # print("early_stage_end", early_stage_end)
            # print("late_stage_start", late_stage_start)
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)


                # 동적 가중치 계산
                timestep_ratio = t.item() / max_timestep
                if timestep_ratio < late_stage_start:
                    # 후기 단계
                    clf_free_weight =  guidance_scale #late_clf_free_weight 
                    classifier_weight = 0
                elif timestep_ratio >= early_stage_end:
                    # 초기 단계
                    clf_free_weight =0# guidance_scale
                    classifier_weight = classifier_guidance_scale  # early_clf_free_weight
                else:
                    # 중간 단계 (선형 변화)
                    progress = (late_stage_start - timestep_ratio) / (late_stage_start - early_stage_end)
                    clf_free_weight = guidance_scale * ( 1- progress) #(progress) * (guidance_scale/2)+guidance_scale/2
                    classifier_weight = classifier_guidance_scale * (progress)




                # Classifier guidance
                if self.classifier is not None and class_labels is not None and gradient_scale!=0:
                    # print("Do Classifier Guidance!")
                    with torch.enable_grad():
                        # print("t", t)
                        latents = latents.detach().requires_grad_()
                        class_scores = self.classifier(latents, t/max_timestep)#, 
                        # print("t/max_timestep", t/max_timestep)

                        predicted_classes = class_scores.argmax(dim=1)
                        correct_predictions = (predicted_classes == class_labels).float()
                        accuracy = correct_predictions.sum() / len(class_labels)
                        
                        # 클래스 변수 업데이트
                        self.__class__.total_accuracy += accuracy.item()
                        self.__class__.accuracy_count += 1
                        
                        # 매 10 스텝마다 평균 정확도 출력 (또는 원하는 간격으로 조정)
                        if self.__class__.accuracy_count % 500 == 0 and self.__class__.accuracy_count > 0:
                            avg_accuracy = self.__class__.total_accuracy / self.__class__.accuracy_count
                            print(f"Average Classifier Accuracy (Count {self.__class__.accuracy_count}): {avg_accuracy * 100:.2f}%")

                        class_loss = -torch.log_softmax(class_scores, dim=1).gather(1, class_labels.unsqueeze(1)).squeeze()
                        grad = torch.autograd.grad(class_loss.sum(), latents)[0]
                    with torch.no_grad():

                        noise_pred_uncond_mean, noise_pred_uncond_std = torch.mean(noise_pred_uncond), torch.std(noise_pred_uncond)
                        noise_pred_text_mean, noise_pred_text_std = torch.mean(noise_pred_text), torch.std(noise_pred_text)
                        # 2. gradient_scale * (1-self.scheduler.alphas_cumprod[t]).sqrt() * grad 값을 더함
                        noise_pred_uncond_1 = noise_pred_uncond + classifier_weight * (1 - self.scheduler.alphas_cumprod[t]).sqrt() * grad
                        noise_pred_text_1 = noise_pred_text + classifier_weight * (1 - self.scheduler.alphas_cumprod[t]).sqrt() * grad

                        # 3. Rescale 단계: 원래 noise_pred의 평균과 표준편차로 다시 맞춤
                        noise_pred_rescaled = (noise_pred_uncond_1 - torch.mean(noise_pred_uncond_1)) / torch.std(noise_pred_uncond_1) * noise_pred_uncond_std + noise_pred_uncond_mean
                        noise_pred_text_rescaled = (noise_pred_text_1 - torch.mean(noise_pred_text_1)) / torch.std(noise_pred_text_1) * noise_pred_text_std + noise_pred_text_mean
                        noise_pred = noise_pred_rescaled
                        # noise_pred = noise_pred_uncond_1
                        latents = latents.detach()

                # Classifier Free guidance
                if do_classifier_free_guidance:
                    noise_pred = noise_pred + clf_free_weight * (noise_pred_text - noise_pred_uncond)
                    # noise_pred = noise_pred_uncond + clf_free_weight * (noise_pred_text - noise_pred_uncond)
                    # noise_pred = noise_pred + clf_free_weight * (noise_pred_text_rescaled - noise_pred_rescaled)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)


        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def set_classifier(self, classifier):
        self.classifier = classifier

    def set_class_to_label(self, class_to_label):
        self.class_to_label = class_to_label

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents