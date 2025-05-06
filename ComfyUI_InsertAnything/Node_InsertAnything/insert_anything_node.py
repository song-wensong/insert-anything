import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
import math


def create_highlighted_mask(image_np, mask_np, alpha=0.5, gray_value=128):


    if mask_np.max() <= 1.0:
        mask_np = (mask_np * 255).astype(np.uint8)
    mask_bool = mask_np > 128

    image_float = image_np.astype(np.float32)

    # 灰色图层
    gray_overlay = np.full_like(image_float, gray_value, dtype=np.float32)

    # 混合
    result = image_float.copy()
    result[mask_bool] = (
        (1 - alpha) * image_float[mask_bool] + alpha * gray_overlay[mask_bool]
    )

    return result.astype(np.uint8)

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)

# Get the bounding box of the mask
def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)

# Expand the bounding box
def expand_bbox(mask, yyxx, ratio, min_crop=0):
    y1,y2,x1,x2 = yyxx
    H,W = mask.shape[0], mask.shape[1]

    yyxx_area = (y2-y1+1) * (x2-x1+1)
    r1 = yyxx_area / (H * W)
    r2 = f(r1)
    ratio = math.sqrt(r2 / r1)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Pad the image to a square shape
def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if len(image.shape) == 2: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    elif len(image.shape) == 3: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)

    return image

# Expand the image and mask
def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask

# Convert the bounding box to a square shape
def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Crop the predicted image back to the original image
def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 2 # maigin_pixel

    if W1 == H1:
        if m != 0:
            tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        else:
            tar_image[y1 :y2, x1:x2, :] =  pred[:, :]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    if m != 0:
        gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    else:
        gen_image[y1 :y2, x1:x2, :] =  pred[:, :]
    
    return gen_image


class MaskOption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "sketch": ("MASK", ),
                "upload": ("MASK", ),
                "option": (["sketch", "upload"], {"default": "sketch"}),

            }
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "MaskOption"

    def MaskOption(self, sketch, upload, option):

        if option == "sketch":
            mask = sketch
        elif option == "upload":
            mask = upload

        return mask


class InsertanythingLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "fill_model_path": ("STRING", {"default": "/path/to/black-forest-labs-FLUX.1-Fill-dev"}),
                "redux_model_path": ("STRING", {"default": "/path/to/black-forest-labs-FLUX.1-Redux-dev"}),
                "lora_path": ("STRING", {"default": "/path/to/lora"}),
                "dtype": (["bfloat16", "bfloat32"], {"default": "bfloat16"}),

            }
        }

    RETURN_TYPES = ("FLUX_FILL_MODEL", "FLUX_REDUX_MODEL")
    FUNCTION = "load_models"

    def load_models(self, fill_model_path, redux_model_path, lora_path, dtype):
        dtype = torch.bfloat16 if dtype == "bfloat16" else torch.bfloat32
        
        # 加载Fill模型
        fill_pipe = FluxFillPipeline.from_pretrained(
            fill_model_path,
            torch_dtype=dtype
        ).to("cuda")
        fill_pipe.load_lora_weights(lora_path)
        

        # 加载Prior模型
        redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            redux_model_path
        ).to(dtype=dtype).to("cuda")
        

        return (fill_pipe, redux_pipe)

class InsertanythingImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {

                "source_image": ("IMAGE", ),
                "ref_image": ("IMAGE", ),
                "source_mask": ("MASK", ),
                "ref_mask": ("MASK", ),
                "redux_pipe": ("FLUX_REDUX_MODEL",),
                "iterations": ("INT", {"default": 2}),

            }
        }
    RETURN_TYPES = (
        "diptych_ref_tar", "mask_diptych", "pipe_prior_output", "H1", "W1", "H2", "W2", "old_tar_image", "tar_box_yyxx_crop", "IMAGE",
    )
    FUNCTION = "process_images"

    def process_images(self, source_image, ref_image, source_mask, ref_mask, redux_pipe, iterations):

        size = (768, 768)


        ref_image = np.array(ref_image)[0]
        tar_image = np.array(source_image)[0]

        ref_image = (ref_image * 255).round().astype(np.uint8)
        tar_image = (tar_image * 255).round().astype(np.uint8)

        ref_mask = np.array(ref_mask)
        source_mask = np.array(source_mask)


        ref_mask = (ref_mask).astype(np.uint8)
        tar_mask = (source_mask).astype(np.uint8)


        if ref_mask.ndim == 3 and ref_mask.shape[2] == 3:
            ref_mask = ref_mask[:, :, 0]

        if tar_mask.ndim == 3 and tar_mask.shape[2] == 3:
            tar_mask = tar_mask[:, :, 0]

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
        pipe_prior_output = redux_pipe(Image.fromarray(masked_ref_image))

        tar_image = pad_to_square(tar_image, pad_value=255)
        H2, W2 = tar_image.shape[0], tar_image.shape[1]

        tar_image = cv2.resize(tar_image, size)
        diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)


        tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
        mask_black = np.ones_like(tar_image) * 0
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

        show_diptych_ref_tar = create_highlighted_mask(diptych_ref_tar, mask_diptych)
        # show_diptych_ref_tar = diptych_ref_tar * (1 - mask_diptych)

        show_diptych_ref_tar = torch.from_numpy(show_diptych_ref_tar).unsqueeze(0).float() / 255.0

        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych = Image.fromarray(mask_diptych)


        return (diptych_ref_tar, mask_diptych, pipe_prior_output, H1, W1, H2, W2, old_tar_image, tar_box_yyxx_crop, show_diptych_ref_tar)

class InsertanythingInferencer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill_pipe": ("FLUX_FILL_MODEL",),
                "diptych_ref_tar": ("diptych_ref_tar",),
                "mask_diptych": ("mask_diptych",),
                "pipe_prior_output": ("pipe_prior_output",),  
                "H1": ("H1",),
                "W1": ("W1",),
                "H2": ("H2",),
                "W2": ("W2",),
                "old_tar_image": ("old_tar_image",),
                "tar_box_yyxx_crop": ("tar_box_yyxx_crop",),
                "seed": ("INT", {"default": 666}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "infer"

    def infer(
        self,
        fill_pipe,
        diptych_ref_tar,
        mask_diptych,
        pipe_prior_output,
        H1, W1, H2, W2,
        old_tar_image,
        seed,
        tar_box_yyxx_crop
    ):
        generator = torch.Generator("cuda").manual_seed(seed)


        edited_image = fill_pipe(
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

        edited_image = torch.from_numpy(edited_image).unsqueeze(0).float() / 255.0

        return (edited_image,)
