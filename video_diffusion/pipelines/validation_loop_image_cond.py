import os
import numpy as  np
from typing import List, Union
import PIL


import torch
import torch.utils.data
import torch.utils.checkpoint

from diffusers.pipeline_utils import DiffusionPipeline
from tqdm.auto import tqdm
from video_diffusion.common.image_util import make_grid, annotate_image
from video_diffusion.common.image_util import save_gif_mp4_folder_type


def load_image_from_path(image_path, height, width, device):
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

class SampleLogger:
    def __init__(
        self,
        editing_prompts: List[str],
        clip_length: int,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        sample_seeds: List[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7,
        editing_images: List[str]=None,
        strength: float = None,
        annotate: bool = True,
        annotate_size: int = 15,
        make_grid: bool = True,
        grid_column_size: int = 2,
        prompt2prompt_edit: bool=False,
        **args
        
    ) -> None:
        self.editing_prompts = editing_prompts
        self.editing_images = editing_images
        self.clip_length = clip_length
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        
        if sample_seeds is None:
            max_num_samples_per_prompt = int(1e5)
            if num_samples_per_prompt > max_num_samples_per_prompt:
                raise ValueError
            sample_seeds = torch.randint(0, max_num_samples_per_prompt, (num_samples_per_prompt,))
            sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds

        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)

        self.annotate = annotate
        self.annotate_size = annotate_size
        self.make_grid = make_grid
        self.grid_column_size = grid_column_size
        self.prompt2prompt_edit = prompt2prompt_edit

    def log_sample_images(
        self, pipeline: DiffusionPipeline,
        device: torch.device, step: int,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latents: torch.FloatTensor = None,
        uncond_embeddings_list: List[torch.FloatTensor] = None,
        cond_prior_embedding: torch.FloatTensor = None,
        source_image_index: int = 0
    ):
        torch.cuda.empty_cache()
        samples_all = []
        attention_all = []
        # handle input image
        if image is not None:
            input_pil_images = pipeline.numpy_to_pil(tensor_to_numpy(image))[0]
            samples_all.append([
                            annotate_image(image, "input sequence", font_size=self.annotate_size) for image in input_pil_images
                        ])
        if self.editing_images is not None:
            for idx, image_path in enumerate(tqdm(self.editing_images, desc="Generating sample images from images")):
                
                
                if self.prompt2prompt_edit:
                    if idx == 0:
                        edit_type = 'save'
                    else:
                        edit_type = 'swap'
                else:
                    edit_type = None
                for seed in self.sample_seeds:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)
                    sequence_return = pipeline(
                        prompt=None,
                        edit_type = edit_type,
                        image=image, # torch.Size([8, 3, 512, 512])
                        strength=self.strength,
                        generator=generator,
                        num_inference_steps=self.num_inference_steps,
                        clip_length=self.clip_length,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=1,
                        edit_image=image_path,
                        cond_prior_embedding=cond_prior_embedding,
                        source_image_index=source_image_index,
                        # used in null inversion
                        latents = latents,
                        uncond_embeddings_list = uncond_embeddings_list,
                        # Put the source prompt at the first one, when using p2p
                    )
                    if self.prompt2prompt_edit:
                        sequence = sequence_return['sdimage_output'].images[0]
                        attention_output = sequence_return['attention_output']
                        if ddim_latents_all_step in sequence_return:
                            ddim_latents_all_step = sequence_return['ddim_latents_all_step']
                    else:
                        sequence = sequence_return.images[0]
                    torch.cuda.empty_cache()

                    anotate_name= f"{image_path.split('/')[-1].split('.')[0]}"
                    if self.annotate:
                        images = [
                            annotate_image(image, anotate_name, font_size=self.annotate_size) for image in sequence
                        ]

                    if self.make_grid:
                        samples_all.append(images)
                        if self.prompt2prompt_edit:
                            attention_all.append(attention_output)
                    save_path = os.path.join(self.logdir, f"step_im_{step}_{idx}_{seed}.gif")
                    save_gif_mp4_folder_type(images, save_path)
                    if self.prompt2prompt_edit:
                        save_gif_mp4_folder_type(attention_output, save_path.replace('.gif', 'atten.gif'))
        
        if self.make_grid:
            samples_all = [make_grid(images, cols=int(np.ceil(np.sqrt(len(samples_all))))) for images in zip(*samples_all)]
            save_path = os.path.join(self.logdir, f"step_im_{step}.gif")
            save_gif_mp4_folder_type(samples_all, save_path)
            if self.prompt2prompt_edit:
                attention_all = [make_grid(images, cols=1) for images in zip(*attention_all)]
                save_gif_mp4_folder_type(attention_all, save_path.replace('.gif', 'atten.gif'))
        
        samples_all = []
        attention_all = []
        
        if self.editing_prompts is not None:
            for idx, prompt in enumerate(tqdm(self.editing_prompts, desc="Generating sample images")):
                if self.prompt2prompt_edit:
                    if idx == 0:
                        edit_type = 'save'
                    else:
                        edit_type = 'swap'
                else:
                    edit_type = None
                for seed in self.sample_seeds:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)
                    sequence_return = pipeline(
                        prompt=prompt,
                        edit_type = edit_type,
                        image=image, # torch.Size([8, 3, 512, 512])
                        strength=self.strength,
                        generator=generator,
                        num_inference_steps=self.num_inference_steps,
                        clip_length=self.clip_length,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=1,
                        # used in null inversion
                        latents = latents,
                        uncond_embeddings_list = uncond_embeddings_list,
                        # Put the source prompt at the first one, when using p2p
                    )
                    if self.prompt2prompt_edit:
                        sequence = sequence_return['sdimage_output'].images[0]
                        attention_output = sequence_return['attention_output']
                        if ddim_latents_all_step in sequence_return:
                            ddim_latents_all_step = sequence_return['ddim_latents_all_step']
                    else:
                        sequence = sequence_return.images[0]
                    torch.cuda.empty_cache()

                    if self.annotate:
                        images = [
                            annotate_image(image, prompt, font_size=self.annotate_size) for image in sequence
                        ]

                    if self.make_grid:
                        samples_all.append(images)
                        if self.prompt2prompt_edit:
                            attention_all.append(attention_output)
                    save_path = os.path.join(self.logdir, f"step_{step}_{idx}_{seed}.gif")
                    save_gif_mp4_folder_type(images, save_path)
                    if self.prompt2prompt_edit:
                        save_gif_mp4_folder_type(attention_output, save_path.replace('.gif', 'atten.gif'))
        
        if self.make_grid:
            samples_all = [make_grid(images, cols=int(np.ceil(np.sqrt(len(samples_all))))) for images in zip(*samples_all)]
            save_path = os.path.join(self.logdir, f"step_{step}.gif")
            save_gif_mp4_folder_type(samples_all, save_path)
            if self.prompt2prompt_edit:
                attention_all = [make_grid(images, cols=1) for images in zip(*attention_all)]
                save_gif_mp4_folder_type(attention_all, save_path.replace('.gif', 'atten.gif'))
        return samples_all


from einops import rearrange

def tensor_to_numpy(image, b=1):
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

    image = image.cpu().float().numpy()
    image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
    return image