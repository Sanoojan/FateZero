import os

import numpy as np
from PIL import Image
from einops import rearrange
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

from .transform import short_size_scale, random_crop, center_crop, offset_crop
from ..common.image_util import IMAGE_EXTENSION
import natsort  
import albumentations as A
import torchvision

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_ids: torch.Tensor,
        prompt: str,
        start_sample_frame: int=0,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = -1, # only used during tuning to sample a long video
        image_mode: str = "RGB",
        image_size: int = 512,
        crop: str = "center",
                
        class_data_root: str = None,
        class_prompt_ids: torch.Tensor = None,
        
        offset: dict = {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0
        },
        **args
        
    ):
        self.path = path
        self.images = self.get_image_list(path)
        self.n_images = len(self.images)
        self.offset = offset
        self.start_sample_frame = start_sample_frame
        if n_sample_frame < 0:
            n_sample_frame = len(self.images)        
        self.n_sample_frame = n_sample_frame
        # local sampling rate from the video
        self.sampling_rate = sampling_rate

        self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if self.n_images < self.sequence_length:
            raise ValueError(f"self.n_images  {self.n_images } < self.sequence_length {self.sequence_length}: Required number of frames {self.sequence_length} larger than total frames in the dataset {self.n_images }")
        
        # During tuning if video is too long, we sample the long video every self.stride globally
        self.stride = stride if stride > 0 else (self.n_images+1)
        self.video_len = (self.n_images - self.sequence_length) // self.stride + 1

        self.image_mode = image_mode
        self.image_size = image_size
        crop_methods = {
            "center": center_crop,
            "random": random_crop,
        }
        if crop not in crop_methods:
            raise ValueError
        self.crop = crop_methods[crop]

        self.prompt = prompt
        self.prompt_ids = prompt_ids
        # Negative prompt for regularization to avoid overfitting during one-shot tuning
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = natsort.natsorted(list(self.class_data_root.iterdir()))
            self.num_class_images = len(self.class_images_path)
            self.class_prompt_ids = class_prompt_ids
        
        
    def __len__(self):
        max_len = (self.n_images - self.sequence_length) // self.stride + 1
        
        if hasattr(self, 'num_class_images'):
            max_len = max(max_len, self.num_class_images)
        
        return max_len

    def __getitem__(self, index):
        return_batch = {}
        frame_indices = self.get_frame_indices(index%self.video_len)
        frames = [self.load_frame(i) for i in frame_indices]
        frames = self.transform(frames)

        return_batch.update(
            {
            "images": frames,
            "prompt_ids": self.prompt_ids,
            }
        )

        if hasattr(self, 'class_data_root'):
            class_index = index % (self.num_class_images - self.n_sample_frame)
            class_indices = self.get_class_indices(class_index)           
            frames = [self.load_class_frame(i) for i in class_indices]
            return_batch["class_images"] = self.tensorize_frames(frames)
            return_batch["class_prompt_ids"] = self.class_prompt_ids
        return return_batch
    
    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = offset_crop(frames, **self.offset)
        frames = short_size_scale(frames, size=self.image_size)
        frames = self.crop(frames, height=self.image_size, width=self.image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1

    def load_frame(self, index):
        image_path = os.path.join(self.path, self.images[index])
        return Image.open(image_path).convert(self.image_mode)

    def load_class_frame(self, index):
        image_path = self.class_images_path[index]
        return Image.open(image_path).convert(self.image_mode)

    def get_frame_indices(self, index):
        if self.start_sample_frame is not None:
            frame_start = self.start_sample_frame + self.stride * index
        else:
            frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    def get_class_indices(self, index):
        frame_start = index
        return (frame_start + i  for i in range(self.n_sample_frame))

    @staticmethod
    def get_image_list(path):
        images = []
        for file in natsort.natsorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images


class ImageSequenceDataset2(Dataset):
    def __init__(
        self,
        path: str,
        prompt_ids: torch.Tensor,
        prompt: str,
        start_sample_frame: int=0,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = -1, # only used during tuning to sample a long video
        image_mode: str = "RGB",
        image_size: int = 512,
        clip_image_size: int = 224,
        crop: str = "center",
                
        class_data_root: str = None,
        class_prompt_ids: torch.Tensor = None,
        
        offset: dict = {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0
        },
        **args
        
    ):
        self.path = path
        self.images = self.get_image_list(path)
        self.n_images = len(self.images)
        self.offset = offset
        self.start_sample_frame = start_sample_frame
        if n_sample_frame < 0:
            n_sample_frame = len(self.images)        
        self.n_sample_frame = n_sample_frame
        # local sampling rate from the video
        self.sampling_rate = sampling_rate

        self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if self.n_images < self.sequence_length:
            raise ValueError(f"self.n_images  {self.n_images } < self.sequence_length {self.sequence_length}: Required number of frames {self.sequence_length} larger than total frames in the dataset {self.n_images }")
        
        # During tuning if video is too long, we sample the long video every self.stride globally
        self.stride = stride if stride > 0 else (self.n_images+1)
        self.video_len = (self.n_images - self.sequence_length) // self.stride + 1

        self.image_mode = image_mode
        self.image_size = image_size
        self.clip_image_size = clip_image_size
        crop_methods = {
            "center": center_crop,
            "random": random_crop,
        }
        if crop not in crop_methods:
            raise ValueError
        self.crop = crop_methods[crop]

        self.prompt = prompt
        self.prompt_ids = prompt_ids
        # Negative prompt for regularization to avoid overfitting during one-shot tuning
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = natsort.natsorted(list(self.class_data_root.iterdir()))
            self.num_class_images = len(self.class_images_path)
            self.class_prompt_ids = class_prompt_ids
        
        
    def __len__(self):
        max_len = (self.n_images - self.sequence_length) // self.stride + 1
        
        if hasattr(self, 'num_class_images'):
            max_len = max(max_len, self.num_class_images)
        
        return max_len

    def __getitem__(self, index):
        return_batch = {}
        frame_indices = self.get_frame_indices(index%self.video_len)
        frames = [self.load_frame(i) for i in frame_indices]
        clip_frames= frames.copy()
        frames = self.transform(frames)
        
        clip_frames = self.clip_transform(clip_frames)
        
        return_batch.update(
            {
            "images": frames,
            "prompt_ids": self.prompt_ids,
            "cond_images": clip_frames
            }
        )

        if hasattr(self, 'class_data_root'):
            class_index = index % (self.num_class_images - self.n_sample_frame)
            class_indices = self.get_class_indices(class_index)           
            frames = [self.load_class_frame(i) for i in class_indices]
            return_batch["class_images"] = self.tensorize_frames(frames)
            return_batch["class_prompt_ids"] = self.class_prompt_ids
            return_batch["class_cond_images"] = self.clip_tensorize_frames(clip_frames)
        return return_batch
    
    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = offset_crop(frames, **self.offset)
        frames = short_size_scale(frames, size=self.image_size)
        frames = self.crop(frames, height=self.image_size, width=self.image_size)
        return frames
    
    def clip_transform(self, frames):
        frames = self.clip_tensorize_frames(frames)
        frames = short_size_scale(frames, size=self.clip_image_size)
        frames = self.crop(frames, height=self.clip_image_size, width=self.clip_image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1
    
    @staticmethod
    def clip_tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        # normalize with (0.48145466, 0.4578275, 0.40821073),
                                                # (0.26862954, 0.26130258, 0.27577711)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1, 1, 1)
        frames = torch.from_numpy(frames).div(255)
        frames = (frames - mean) / std
        return frames

    def load_frame(self, index):
        image_path = os.path.join(self.path, self.images[index])
        return Image.open(image_path).convert(self.image_mode)

    def load_class_frame(self, index):
        image_path = self.class_images_path[index]
        return Image.open(image_path).convert(self.image_mode)

    def get_frame_indices(self, index):
        if self.start_sample_frame is not None:
            frame_start = self.start_sample_frame + self.stride * index
        else:
            frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    def get_class_indices(self, index):
        frame_start = index
        return (frame_start + i  for i in range(self.n_sample_frame))

    @staticmethod
    def get_image_list(path):
        images = []
        for file in natsort.natsorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images



class ImageSequenceDatasetMulti(Dataset):
    def __init__(
        self,
        path: str,  # List of directories instead of a single path
        prompt_ids: torch.Tensor,
        prompt: str,
        start_sample_frame: int=0,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = -1, 
        image_mode: str = "RGB",
        image_size: int = 512,
        clip_image_size: int = 224,
        crop: str = "center",
                
        class_data_root: str = None,
        class_prompt_ids: torch.Tensor = None,
        
        offset: dict = {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0
        },
        **args
    ):
        
        self.paths= [ os.path.join(path, folder) for folder in os.listdir(path)]
        # self.paths = paths  # List of folder paths
        self.image_folders = [self.get_image_list(path) for path in self.paths]  # Get images from all folders
        self.n_images_in_folders = [len(images) for images in self.image_folders]  # Store the number of images in each folder

        self.total_videos = len(self.paths)  # Number of folders
        self.offset = offset
        self.start_sample_frame = start_sample_frame
        self.n_sample_frame = n_sample_frame
        self.sampling_rate = sampling_rate
        self.image_mode=image_mode

        self.random_trans=A.Compose([
            A.Resize(height=512,width=512),
            A.RandomCrop(height=400,width=400),
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3), 
            # A.GaussNoise(p=0.3),# newly added from this line
            # A.HueSaturationValue(p=0.3),
            # A.ISONoise(p=0.3),
            # A.Solarize(p=0.3),
            ])

        # Define stride behavior (using stride to sample long videos)
        self.stride = stride if stride > 0 else max(self.n_images_in_folders)

        # Cropping method (center/random)
        crop_methods = {
            "center": center_crop,
            "random": random_crop,
        }
        if crop not in crop_methods:
            raise ValueError
        self.crop = crop_methods[crop]

        self.prompt = prompt
        self.prompt_ids = prompt_ids
        self.image_size = image_size
        self.clip_image_size = clip_image_size
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = natsort.natsorted(list(self.class_data_root.iterdir()))
            self.num_class_images = len(self.class_images_path)
            self.class_prompt_ids = class_prompt_ids

    def __len__(self):
        max_len = max([(n - self.n_sample_frame) // self.stride + 1 for n in self.n_images_in_folders])
        if hasattr(self, 'num_class_images'):
            max_len = max(max_len, self.num_class_images)
        return max_len * self.total_videos

    def __getitem__(self, index):
        folder_index = index // self.total_videos  # Get folder index
        video_index = index % self.total_videos  # Get video index within the folder

        return_batch = {}
        frame_indices = self.get_frame_indices(video_index, folder_index)
        frames = [self.load_frame(i, folder_index) for i in frame_indices]
        # clip_frames = frames.copy()

        frames = self.transform(frames)
        # if frames.shape[-2]!=512:
        #     print("Error")
        #     breakpoint()
        # select any random image in folder index
        sel_num=np.random.randint(0,self.n_images_in_folders[folder_index])
        sel_path=os.path.join(self.paths[folder_index], self.image_folders[folder_index][sel_num])
        img_p_np=cv2.imread(sel_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_img=self.random_trans(image=img_p_np)["image"]
        ref_image_tensor=Image.fromarray(ref_img)
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)
        # clip_frames = self.clip_transform(clip_frames)
        
        return_batch.update(
            {
                "images": frames,
                "prompt_ids": self.prompt_ids,
                "cond_images": ref_image_tensor,
                "folder": self.paths[folder_index]
            }
        )

        if hasattr(self, 'class_data_root'):
            class_index = index % (self.num_class_images - self.n_sample_frame)
            class_indices = self.get_class_indices(class_index)
            frames = [self.load_class_frame(i) for i in class_indices]
            return_batch["class_images"] = self.tensorize_frames(frames)
            return_batch["class_prompt_ids"] = self.class_prompt_ids
            return_batch["class_cond_images"] = self.clip_tensorize_frames(ref_image_tensor)

        return return_batch
    
    def get_frame_indices(self, index, folder_index):
        """Get frame indices for the current folder and video, ensuring they stay within bounds."""
        n_images = self.n_images_in_folders[folder_index]  # Get number of images in the current folder

        # if self.start_sample_frame is not None:
        #     frame_start = self.start_sample_frame + self.stride * index
        # else:
        #     frame_start = self.stride * index
        
        frame_start=index
        if frame_start >= n_images-self.n_sample_frame:
            frame_start = n_images-self.n_sample_frame
        
        # breakpoint()
        # Ensure frame indices do not exceed the number of images in the folder
        return [
            min(frame_start + i * self.sampling_rate, n_images - 1)  # Cap the index to the max available image
            for i in range(self.n_sample_frame)
        ]

    def load_frame(self, index, folder_index):
        """Load a frame from the specific folder."""
        image_path = os.path.join(self.paths[folder_index], self.image_folders[folder_index][index])
        return Image.open(image_path).convert(self.image_mode)

    @staticmethod
    def get_image_list(path):
        """Get a list of image files in a folder."""
        images = []
        for file in natsort.natsorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images
    
    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = offset_crop(frames, **self.offset)
        frames = short_size_scale(frames, size=self.image_size)
        frames = self.crop(frames, height=self.image_size, width=self.image_size)
        return frames
    
    def clip_transform(self, frames):
        frames = self.clip_tensorize_frames(frames)
        frames = short_size_scale(frames, size=self.clip_image_size)
        frames = self.crop(frames, height=self.clip_image_size, width=self.clip_image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1
    
    @staticmethod
    def clip_tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        # normalize with (0.48145466, 0.4578275, 0.40821073),
                                                # (0.26862954, 0.26130258, 0.27577711)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1, 1, 1)
        frames = torch.from_numpy(frames).div(255)
        frames = (frames - mean) / std
        return frames
    
# class ImageSequenceDatasetMulti(Dataset):
#     def __init__(
#         self,
#         path: str,
#         prompt_ids: torch.Tensor,
#         prompt: str,
#         start_sample_frame: int=0,
#         n_sample_frame: int = 8,
#         sampling_rate: int = 1,
#         stride: int = -1, # only used during tuning to sample a long video
#         image_mode: str = "RGB",
#         image_size: int = 512,
#         clip_image_size: int = 224,
#         crop: str = "center",
                
#         class_data_root: str = None,
#         class_prompt_ids: torch.Tensor = None,
        
#         offset: dict = {
#             "left": 0,
#             "right": 0,
#             "top": 0,
#             "bottom": 0
#         },
#         **args
        
#     ):
#         self.paths= [ os.path.join(path, folder) for folder in os.listdir(path)]
#         # self.paths = paths  # List of folder paths
#         self.images = [self.get_image_list(path) for path in self.paths]  # Get images from all folders
#         self.n_images = [len(images) for images in self.images]  # Store the number of images in each folder

#         self.total_videos = len(self.paths)  # Number of folders
        
#         # self.images = self.get_image_list(path)
#         # self.n_images = len(self.images)
#         self.offset = offset
#         self.start_sample_frame = start_sample_frame
#         if n_sample_frame < 0:
#             n_sample_frame = len(self.images)        
#         self.n_sample_frame = n_sample_frame
#         # local sampling rate from the video
#         self.sampling_rate = sampling_rate

#         self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
#         if self.n_images < self.sequence_length:
#             raise ValueError(f"self.n_images  {self.n_images } < self.sequence_length {self.sequence_length}: Required number of frames {self.sequence_length} larger than total frames in the dataset {self.n_images }")
        
#         # During tuning if video is too long, we sample the long video every self.stride globally
#         self.stride = stride if stride > 0 else (self.n_images+1)
#         self.video_len = (self.n_images - self.sequence_length) // self.stride + 1

#         self.image_mode = image_mode
#         self.image_size = image_size
#         self.clip_image_size = clip_image_size
#         crop_methods = {
#             "center": center_crop,
#             "random": random_crop,
#         }
#         if crop not in crop_methods:
#             raise ValueError
#         self.crop = crop_methods[crop]

#         self.prompt = prompt
#         self.prompt_ids = prompt_ids
#         # Negative prompt for regularization to avoid overfitting during one-shot tuning
#         if class_data_root is not None:
#             self.class_data_root = Path(class_data_root)
#             self.class_images_path = natsort.natsorted(list(self.class_data_root.iterdir()))
#             self.num_class_images = len(self.class_images_path)
#             self.class_prompt_ids = class_prompt_ids
        
        
#     def __len__(self):
#         max_len = (self.n_images - self.sequence_length) // self.stride + 1
        
#         if hasattr(self, 'num_class_images'):
#             max_len = max(max_len, self.num_class_images)
        
#         return max_len

#     def __getitem__(self, index):
#         return_batch = {}
#         frame_indices = self.get_frame_indices(index%self.video_len)
#         frames = [self.load_frame(i) for i in frame_indices]
#         clip_frames= frames.copy()
#         frames = self.transform(frames)
        
#         clip_frames = self.clip_transform(clip_frames)
        
#         return_batch.update(
#             {
#             "images": frames,
#             "prompt_ids": self.prompt_ids,
#             "cond_images": clip_frames
#             }
#         )

#         if hasattr(self, 'class_data_root'):
#             class_index = index % (self.num_class_images - self.n_sample_frame)
#             class_indices = self.get_class_indices(class_index)           
#             frames = [self.load_class_frame(i) for i in class_indices]
#             return_batch["class_images"] = self.tensorize_frames(frames)
#             return_batch["class_prompt_ids"] = self.class_prompt_ids
#             return_batch["class_cond_images"] = self.clip_tensorize_frames(clip_frames)
#         return return_batch
    
#     def transform(self, frames):
#         frames = self.tensorize_frames(frames)
#         frames = offset_crop(frames, **self.offset)
#         frames = short_size_scale(frames, size=self.image_size)
#         frames = self.crop(frames, height=self.image_size, width=self.image_size)
#         return frames
    
#     def clip_transform(self, frames):
#         frames = self.clip_tensorize_frames(frames)
#         frames = short_size_scale(frames, size=self.clip_image_size)
#         frames = self.crop(frames, height=self.clip_image_size, width=self.clip_image_size)
#         return frames

#     @staticmethod
#     def tensorize_frames(frames):
#         frames = rearrange(np.stack(frames), "f h w c -> c f h w")
#         return torch.from_numpy(frames).div(255) * 2 - 1
    
#     @staticmethod
#     def clip_tensorize_frames(frames):
#         frames = rearrange(np.stack(frames), "f h w c -> c f h w")
#         # normalize with (0.48145466, 0.4578275, 0.40821073),
#                                                 # (0.26862954, 0.26130258, 0.27577711)
#         mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1, 1, 1)
#         std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1, 1, 1)
#         frames = torch.from_numpy(frames).div(255)
#         frames = (frames - mean) / std
#         return frames

#     def load_frame(self, index):
#         image_path = os.path.join(self.path, self.images[index])
#         return Image.open(image_path).convert(self.image_mode)

#     def load_class_frame(self, index):
#         image_path = self.class_images_path[index]
#         return Image.open(image_path).convert(self.image_mode)

#     def get_frame_indices(self, index):
#         if self.start_sample_frame is not None:
#             frame_start = self.start_sample_frame + self.stride * index
#         else:
#             frame_start = self.stride * index
#         return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

#     def get_class_indices(self, index):
#         frame_start = index
#         return (frame_start + i  for i in range(self.n_sample_frame))

#     @staticmethod
#     def get_image_list(path):
#         images = []
#         for file in natsort.natsorted(os.listdir(path)):
#             if file.endswith(IMAGE_EXTENSION):
#                 images.append(file)
#         return images