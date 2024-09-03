import os,copy

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import inspect
from typing import Optional, Dict
import click
from omegaconf import OmegaConf
import PIL
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)

from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipeline_utils import DiffusionPipeline

from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from video_diffusion.models.modules import FrozenCLIPImageEmbedder
from einops import rearrange

from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.data.dataset import ImageSequenceDataset, ImageSequenceDataset2
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.logger import get_logger_config_path
from video_diffusion.common.image_util import log_train_samples, log_train_reg_samples
from video_diffusion.common.instantiate_from_config import instantiate_from_config, get_obj_from_str
from video_diffusion.pipelines.validation_loop_image_cond import SampleLogger
from transformers.models.clip.configuration_clip import CLIPConfig,CLIPVisionConfig




def collate_fn(examples):
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
        "cond_images": torch.stack([example["cond_images"] for example in examples]),
    }
    if "class_images" in examples[0]:
        batch["class_prompt_ids"] = torch.cat([example["class_prompt_ids"] for example in examples], dim=0)
        batch["class_images"] =  torch.stack([example["class_images"] for example in examples])
        batch["class_cond_images"] =  torch.stack([example["class_cond_images"] for example in examples])
    return batch


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


def test(
    config: str,
    pretrained_model_path: str,
    dataset_config: Dict,
    logdir: str = None,
    train_steps: int = 300,
    validation_steps: int = 1000,
    editing_config: Optional[Dict] = None,
    test_pipeline_config: Optional[Dict] = dict(),
    trainer_pipeline_config: Optional[Dict] = dict(),
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers: bool = True,
    train_batch_size: int = 1,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_checkpointing: bool = False,
    train_temporal_conv: bool = False,
    checkpointing_steps: int = 1000,
    model_config: dict={},
):
    args = get_function_args()
    train_dataset_config = copy.deepcopy(dataset_config)
    time_string = get_time_string()
    if logdir is None:
        logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
    logdir += f"_{time_string}"


    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        device_placement=True,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))
    logger = get_logger_config_path(logdir)
    if seed is not None:
        set_seed(seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    ) 
    visual_encoder = FrozenCLIPImageEmbedder()
    visual_encoder_weights = torch.load(os.path.join(pretrained_model_path, "visual_encoder", "pytorch_model.bin"))
    visual_encoder.load_state_dict(visual_encoder_weights)

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    )

    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet"), model_config=model_config
    )

    
    if 'target' not in test_pipeline_config:
        test_pipeline_config['target'] = 'video_diffusion.pipelines.stable_diffusion.SpatioTemporalStableDiffusionPipeline'
    
    pipeline = instantiate_from_config(
        test_pipeline_config,
        vae=vae,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        ),
    )
    pipeline.scheduler.set_timesteps(editing_config['num_inference_steps'])
    pipeline.set_progress_bar_config(disable=True)


    if is_xformers_available() and enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print('enable xformers in the training and testing')
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # visual_encoder.requires_grad_(False)  # already require grad is true in the class for mapping


    
                

    if gradient_checkpointing:
        print('enable gradient checkpointing in the training and testing')    
        unet.enable_gradient_checkpointing()


    prompt_ids = tokenizer(
        train_dataset_config["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    
    if 'class_data_root' in train_dataset_config:
        if 'class_data_prompt' not in train_dataset_config:
            train_dataset_config['class_data_prompt'] = train_dataset_config['prompt']
        class_prompt_ids = tokenizer(
            train_dataset_config["class_data_prompt"],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
    else:
        class_prompt_ids = None
    train_dataset = ImageSequenceDataset2(**train_dataset_config, prompt_ids=prompt_ids, class_prompt_ids=class_prompt_ids) # change here for loading more data

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn,
    )

    train_sample_save_path = os.path.join(logdir, "train_samples.gif")
    log_train_samples(save_path=train_sample_save_path, train_dataloader=train_dataloader)
    if 'class_data_root' in train_dataset_config:
        log_train_reg_samples(save_path=train_sample_save_path.replace('train_samples', 'class_data_samples'), train_dataloader=train_dataloader)


    unet,visual_encoder, train_dataloader = accelerator.prepare(
        unet,visual_encoder, train_dataloader
    )
  

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        print('enable float16 in the training and testing')
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # visual_encoder.to(accelerator.device, dtype=weight_dtype)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  # , config=vars(args))
    
 
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    
    step = 0
    # End of config trainer
    
    if editing_config is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**editing_config, logdir=logdir)


    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    
    assert(train_dataset.video_len == 1), "Only support overfiting on a single video"


    batch = next(train_data_yielder)

    if accelerator.sync_gradients:

        if accelerator.is_main_process:
            with torch.no_grad():

                if validation_sample_logger is not None and (step % validation_steps == 0 or step==1):
                    unet.eval()
                    visual_encoder.eval()

                    val_image = rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w")

                    # Unet is changing in different iteration; we should invert online
                    if editing_config.get('use_invertion_latents', False):
                        # Precompute the latents for this video to align the initial latents in training and test
                        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
                        # we only inference for latents, no training
                        vae.eval()
                        text_encoder.eval()
                        visual_encoder.eval()
                        unet.eval()
                        
                        text_embeddings = pipeline._encode_prompt(
                                train_dataset.prompt, 
                                device = accelerator.device, 
                                num_images_per_prompt = 1, 
                                do_classifier_free_guidance = True, 
                                negative_prompt=None
                        )
                        im_path=train_dataset_config['image']
                        edit_image = load_image_from_path(im_path, 224, 224, accelerator.device)
                        visual_embedding = visual_encoder(edit_image.unsqueeze(0))[0]
                        uncond_from_text = text_embeddings[0]
                        visual_embedding = visual_embedding.repeat(uncond_from_text.shape[0], 1)
                        visual_embedding = torch.stack([ uncond_from_text,visual_embedding], dim=0)
                        
                        batch['latents_all_step'] = pipeline.prepare_latents_ddim_inverted(
                            rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"), 
                            batch_size = 1 , 
                            num_images_per_prompt = 1,  # not sure how to use it
                            text_embeddings = text_embeddings
                            )
                        batch['ddim_init_latents'] = batch['latents_all_step'][-1]
                    else:
                        batch['ddim_init_latents'] = None                    
                    
                    
                    
                    validation_sample_logger.log_sample_images(
                        image= val_image, # torch.Size([8, 3, 512, 512])
                        pipeline=pipeline,
                        device=accelerator.device,
                        step=step,
                        latents = batch['ddim_init_latents'], #torch.Size([1, 4, 8, 64, 64])
                    )
                    torch.cuda.empty_cache()
                    unet.train()
                    visual_encoder.train()

                

    accelerator.end_training()


@click.command()
@click.option("--config", type=str, default="config/tune/faceswap_andy_with_im_infer.yaml")
def run(config):
    test(config=config, **OmegaConf.load(config))


if __name__ == "__main__":
    run()
