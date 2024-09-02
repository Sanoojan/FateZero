# code mostly taken from https://github.com/huggingface/diffusers
import inspect
from typing import Callable, List, Optional, Union
import PIL
import torch
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm

from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from .stable_diffusion_image_cond import SpatioTemporalStableDiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_image_from_path(image_path, height, width, device):
        if isinstance(image_path, list):
            images = []
            for i in range(len(image_path)):
                image = PIL.Image.open(image_path[i]).convert("RGB")
                # normalize with (0.48145466, 0.4578275, 0.40821073),
                #                                         # (0.26862954, 0.26130258, 0.27577711)
                image = image.resize((width, height), PIL.Image.LANCZOS)
                image=clip_tensorize_frames(image)
                images.append(image)
            return torch.stack(images).to(device)
                
        image = PIL.Image.open(image_path).convert("RGB")
        # normalize with (0.48145466, 0.4578275, 0.40821073),
        #                                         # (0.26862954, 0.26130258, 0.27577711)
        image = image.resize((width, height), PIL.Image.LANCZOS)
        image=clip_tensorize_frames(image)
        return image.to(device)
        
def clip_transform( frames):
    
    frames = clip_tensorize_frames(frames)
    # frames = short_size_scale(frames, size=self.clip_image_size)
    # frames = self.crop(frames, height=clip_image_size, width=clip_image_size)
    return frames

def clip_tensorize_frames(frames):
    frames = np.array(frames)
    frames = rearrange(np.stack(frames), "h w c -> c h w")
    # normalize with (0.48145466, 0.4578275, 0.40821073),
                                            # (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    frames = torch.from_numpy(frames).div(255)
    frames = (frames - mean) / std
    return frames

class DDIMSpatioTemporalStableDiffusionPipeline(SpatioTemporalStableDiffusionPipeline):  # Do it here @ sanoojan
    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """

    def check_inputs(self, prompt,edit_image, height, width, callback_steps, strength=None):
        if not isinstance(prompt, str) and not isinstance(prompt, list) and edit_image is None:
            raise ValueError(f"`prompt or edit_image` has to be of type `str` or `list` but is {type(prompt)}")
        if strength is not None:
            if strength <= 0 or strength > 1:
                raise ValueError(f"The value of strength should in (0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )



    def prepare_latents_ddim_inverted(self, image, batch_size, num_images_per_prompt, 
                                    #   dtype, device, 
                                      text_embeddings,
                                      generator=None): 
        
        # Not sure if image need to change device and type
        # image = image.to(device=device, dtype=dtype)
        
        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)
        init_latents = 0.18215 * init_latents

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

        # get latents
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)
        ddim_latents_all_step = self.ddim_clean2noisy_loop(init_latents_bcfhw, text_embeddings)
        return ddim_latents_all_step
    
    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings):
        weight_dtype = latent.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        print('Invert clean image to noise latents by DDIM and Unet')
        for i in trange(len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            # noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"] # [1, 4, 8, 64, 64] ->  [1, 4, 8, 64, 64])
            latent = self.next_clean2noise_step(noise_pred, t, latent)
            all_latent.append(latent.to(dtype=weight_dtype))
        
        return all_latent
    
    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """
        Assume the eta in DDIM=0
        """
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]]=None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = None,
        num_inference_steps: int = 50,
        clip_length: int = 8,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        edit_image: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **args
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Only used in DDIM or strength<1.0
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.            
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet

        
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, edit_image,height, width, callback_steps, strength)

        # 2. Define call parameters
        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        elif edit_image is not None:
            batch_size = 1 if isinstance(edit_image, str) else len(edit_image)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt="a photo of a human face"
        # 3. Encode input prompt
        if prompt is not None:
            text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        else:
            text_embeddings = None
            uncond_cond=  self._encode_prompt(
                "Empty", device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )[0]       
        if edit_image is not None:
            # load edit_image
            edit_image = load_image_from_path(edit_image, 224, 224, device)
            uncond_cond=text_embeddings[0]
            visual_embeddings = self.visual_encoder(edit_image.unsqueeze(0))[0]
            visual_embeddings = visual_embeddings.repeat(text_embeddings.shape[1], 1)
            visual_embeddings = torch.stack([ uncond_cond,visual_embeddings], dim=0)
        else:
            visual_embeddings = None
        
        # embedding= text_embeddings if text_embeddings is not None else visual_embeddings
        embedding=visual_embeddings if visual_embeddings is not None else text_embeddings
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # if strength <1.0:
        # timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        timesteps = self.scheduler.timesteps
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        if latents is None:
            ddim_latents_all_step = self.prepare_latents_ddim_inverted(
                image, batch_size, num_images_per_prompt, 
                # text_embeddings.dtype, device, 
                text_embeddings,   # checking normal text embeddings  chage later to embedding @ sanoojan
                generator,
            )
            latents = ddim_latents_all_step[-1]

        latents_dtype = latents.dtype

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # check later @ sanoojan

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=embedding
                ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        has_nsfw_concept = None

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    