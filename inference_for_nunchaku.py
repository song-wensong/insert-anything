from PIL import Image
import torch
import os
import numpy as np
import cv2
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

device = torch.device(f"cuda")
dtype = torch.bfloat16
size = (768, 768)



# Load the pre-trained model and LoRA-for-nunchaku weights
# Please replace the paths with your own paths
transformer = NunchakuFluxTransformer2dModel.from_pretrained("/path/to/svdq-int4-flux.1-fill-dev")

pipe = FluxFillPipeline.from_pretrained(
    "/path/to/black-forest-labs-FLUX.1-Fill-dev",
    transformer=transformer,
    torch_dtype=dtype
)

transformer.update_lora_params("/path/to/lora-for-nunchaku")  
# Adjust the LoRA strength
transformer.set_lora_strength(1)

redux = FluxPriorReduxPipeline.from_pretrained("/path/to/black-forest-labs-FLUX.1-Redux-dev").to(dtype=dtype)



# The purpose of this code is to reduce the GPU memory usage to 26GB, but it will increase the inference time accordingly.
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
redux.enable_model_cpu_offload()


# Load the source and reference images and masks
# Please replace the paths with your own image and mask paths
source_image_path = "examples/source_image/1.png"
mask_image_path = "examples/source_mask/1.png"

ref_image_path = "examples/ref_image/1.png"
ref_mask_path = "examples/ref_mask/1.png"

# Load the images and masks
ref_image = cv2.imread(ref_image_path)
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
tar_image = cv2.imread(source_image_path)
tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]
tar_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]
tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

# Remove the background information of the reference picture
ref_box_yyxx = get_bbox_from_mask(ref_mask)
ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 

# Extract the box where the reference image is located, and place the reference object at the center of the image
y1,y2,x1,x2 = ref_box_yyxx
masked_ref_image = masked_ref_image[y1:y2,x1:x2,:] 
ref_mask = ref_mask[y1:y2,x1:x2] 
ratio = 1.3
masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 
masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)

# Dilate the mask
kernel = np.ones((7, 7), np.uint8)
iterations = 2
tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

# zome in
tar_box_yyxx = get_bbox_from_mask(tar_mask)
tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=2)   
tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
y1,y2,x1,x2 = tar_box_yyxx_crop

old_tar_image = tar_image.copy()
tar_image = tar_image[y1:y2,x1:x2,:]
tar_mask = tar_mask[y1:y2,x1:x2]

H1, W1 = tar_image.shape[0], tar_image.shape[1]

tar_mask = pad_to_square(tar_mask, pad_value=0)
tar_mask = cv2.resize(tar_mask, size)

# Extract the features of the reference image
masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
pipe_prior_output = redux(Image.fromarray(masked_ref_image))

tar_image = pad_to_square(tar_image, pad_value=255)
H2, W2 = tar_image.shape[0], tar_image.shape[1]

tar_image = cv2.resize(tar_image, size)
diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)

tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
mask_black = np.ones_like(tar_image) * 0
mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

diptych_ref_tar = Image.fromarray(diptych_ref_tar)
mask_diptych[mask_diptych == 1] = 255
mask_diptych = Image.fromarray(mask_diptych)

seeds = [666]

for seed in seeds:

    generator = torch.Generator(device).manual_seed(seed)
    edited_image = pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        generator=generator,
        **pipe_prior_output,
    ).images[0]

    width, height = edited_image.size
    left = width // 2
    right = width
    top = 0
    bottom = height
    edited_image = edited_image.crop((left, top, right, bottom))

    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
    edited_image = Image.fromarray(edited_image)

    ref_with_ext = os.path.basename(ref_mask_path)
    tar_with_ext = os.path.basename(mask_image_path)
    ref_without_ext = os.path.splitext(ref_with_ext)[0]
    tar_without_ext = os.path.splitext(tar_with_ext)[0]
    
    save_path = "./result"
    os.makedirs(save_path, exist_ok=True)
    edited_image_save_path = os.path.join(save_path, f"{ref_without_ext}_to_{tar_without_ext}_seed{seed}.png")
    edited_image.save(edited_image_save_path)


