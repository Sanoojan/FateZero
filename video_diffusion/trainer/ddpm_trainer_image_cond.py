from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer
from video_diffusion.models.modules import FrozenCLIPImageEmbedder

from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.pipelines.stable_diffusion_image_cond import SpatioTemporalStableDiffusionPipeline

from Libraries.Face_models.encoders.model_irse import Backbone
import torch.nn as nn
import torchvision.transforms.functional as TF 

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

def un_norm(x):
    return (x+1.0)/2.0

class IDLoss(nn.Module):
    def __init__(self,opts=None,multiscale=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.opts = opts 
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # self.facenet=iresnet100(pretrained=False, fp16=False) # changed by sanoojan
        
        self.facenet.load_state_dict(torch.load("Other_dependencies/arcface/model_ir_se50.pth"))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=True):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )
        
        # x_feats = self.facenet(x) # changed by sanoojan
        return x_feats

class DDPMTrainer(SpatioTemporalStableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        visual_encoder: FrozenCLIPImageEmbedder,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        **kwargs
    ):
        super().__init__(
            vae,
            text_encoder,
            visual_encoder,
            tokenizer,
            unet,
            scheduler,
        )
        self.ID_encoder=IDLoss().to('cuda')
        for name, module in kwargs.items():
            setattr(self, name, module)

    def step(self, 
             batch: dict = dict()):
        if 'class_images' in batch:
            self.step2d(batch["class_images"],batch["class_cond_images"], batch["class_prompt_ids"])
        self.vae.eval()
        self.text_encoder.eval()
        # self.visual_encoder.eval()
        self.unet.train()
        self.visual_encoder.train()
        
        
        if self.prior_preservation is not None:
            print('Use prior_preservation loss')
            self.unet2d.eval()



        # Convert images to latent space
        images = batch["images"].to(dtype=self.weight_dtype)
        # cond_images=rearrange(batch["cond_images"], "b c f h w -> (b f) c h w")
        # cond_images=cond_images.to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        latents = self.vae.encode(images).latent_dist.sample() # shape=torch.Size([8, 3, 512, 512]), min=-1.00, max=0.98, var=0.21, -0.96875
        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b)
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        
        Full_prompt=batch["Full_prompt"]
        # replace _ with " "
        Full_prompt=Full_prompt.replace("_"," ")
        
        prompt_ids = self.tokenizer(
            Full_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = prompt_ids.to(self.device)
        
        tokens=self.tokenizer.tokenize(Full_prompt)
        #find indexe of "person" in Full_prompt  
        person_index=tokens.index("person</w>")+1
        background_index=tokens.index("background</w>")+1
        

        # Get the text embedding for conditioning
        # encoder_hidden_states = self.text_encoder(batch["prompt_ids"])[0] # shape b,77,768
        # batch["cond_images"] shape [1,3,8,512,512] 
        # rearrange to [8,3,512,512]
        
        
        cond_images=batch["cond_images"]  # only one image sample
        random_index=torch.randint(0,cond_images.shape[0],(1,))
        ID_features=self.ID_encoder.extract_feats(cond_images)[0].unsqueeze(0)
        ID_features=self.visual_encoder.ID_proj_out(ID_features)
        
        visual_hidden_states = self.visual_encoder(cond_images) 
        visual_hidden_states=visual_hidden_states[random_index]
        ID_features=(10.0*ID_features+1.0*visual_hidden_states)/11.0
        
     
        background_img=batch["background_img"] 
        background_hidden_states=self.visual_encoder(background_img)
        background_hidden_states=background_hidden_states[random_index]
     
        # visual_hidden_states= visual_hidden_states[0].repeat(77, 1)
        # visual_hidden_states=torch.unsqueeze(visual_hidden_states,0)
       
        encoder_hidden_states = self.text_encoder(prompt_ids)[0]
        #replace the person and background embeddings 
        encoder_hidden_states[:,person_index,:]=ID_features
        encoder_hidden_states[:,background_index,:]=background_hidden_states

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.prior_preservation is not None:
            model_pred_2d = self.unet2d(noisy_latents[:, :, 0], timesteps, encoder_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * self.prior_preservation
            )

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    
    def step2d(self, class_images,cond_images, prompt_ids
             ):
        
        self.vae.eval()
        self.text_encoder.eval()
        # self.visual_encoder.eval()
        self.unet.train()
        self.visual_encoder.train()
        if self.prior_preservation is not None:
            self.unet2d.eval()


        # Convert images to latent space
        images = class_images.to(dtype=self.weight_dtype)
        # cond_images=rearrange(cond_images, "b c f h w -> (b f) c h w")
        cond_images = cond_images.to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        latents = self.vae.encode(images).latent_dist.sample() # shape=torch.Size([8, 3, 512, 512]), min=-1.00, max=0.98, var=0.21, -0.96875
        
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # encoder_hidden_states = self.text_encoder(prompt_ids)[0]
        visual_hidden_states = self.visual_encoder(cond_images)[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, visual_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.prior_preservation is not None:
            model_pred_2d = self.unet2d(noisy_latents[:, :, 0], timesteps, visual_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * self.prior_preservation
            )

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss