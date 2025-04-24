import os
import sys
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageFilter, ImageDraw
from huggingface_hub import snapshot_download
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
import math
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

dtype = torch.bfloat16
size = (768, 768)

pipe = FluxFillPipeline.from_pretrained(
    "/path/to/black-forest-labs-FLUX.1-Fill-dev",
    torch_dtype=dtype
).to("cuda")

pipe.load_lora_weights(
    "/path/to/lora"
)

redux = FluxPriorReduxPipeline.from_pretrained("/path/to/black-forest-labs-FLUX.1-Redux-dev").to(dtype=dtype).to("cuda")

###   example  #####
ref_dir='./examples/ref_image'
ref_mask_dir='./examples/ref_mask'
image_dir='./examples/source_image'
image_mask_dir='./examples/source_mask'

ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
ref_list.sort()

ref_mask_list=[os.path.join(ref_mask_dir,file) for file in os.listdir(ref_mask_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
ref_mask_list.sort()

image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
image_list.sort()

image_mask_list=[os.path.join(image_mask_dir,file) for file in os.listdir(image_mask_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
image_mask_list.sort()
###   example  #####


def run_local(base_image, base_mask, reference_image, ref_mask, seed, base_mask_option, ref_mask_option):


    if base_mask_option == "Draw Mask":
        tar_image = base_image["image"]
        tar_mask = base_image["mask"]
    else:
        tar_image = base_image["image"]
        tar_mask = base_mask

    if ref_mask_option == "Draw Mask":
        ref_image = reference_image["image"]
        ref_mask = reference_image["mask"]
    else:
        ref_image = reference_image["image"]
        ref_mask = ref_mask


    tar_image = tar_image.convert("RGB")
    tar_mask = tar_mask.convert("L")
    ref_image = ref_image.convert("RGB")
    ref_mask = ref_mask.convert("L")

    tar_image = np.asarray(tar_image)
    tar_mask = np.asarray(tar_mask)
    tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)


    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2] 
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)


    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 

    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # zome in
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=2)    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop


    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2,x1:x2,:]
    tar_mask = tar_mask[y1:y2,x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]
    # zome in


    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

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

    generator = torch.Generator("cuda").manual_seed(seed)
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


    return [edited_image]

def update_ui(option):
    if option == "Draw Mask":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


with gr.Blocks() as demo:

    gr.Markdown("#  Play with InsertAnything to Insert your Target Objects! ")
    gr.Markdown("# Upload / Draw Images for the Background (up) and Reference Object (down)")
    gr.Markdown("### Draw mask on the background or just upload the mask.")
    gr.Markdown("### Only select one of these two methods. Don't forget to click the corresponding button!!")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                base_image = gr.Image(label="Background Image", source="upload", tool="sketch", type="pil",
                                        brush_color='#FFFFFF', mask_opacity=0.5)

                base_mask = gr.Image(label="Background Mask", source="upload", type="pil")

            with gr.Row():
                base_mask_option = gr.Radio(["Draw Mask", "Upload with Mask"], label="Background Mask Input Option", value="Upload with Mask")

            with gr.Row():
                ref_image = gr.Image(label="Reference Image", source="upload", tool="sketch", type="pil", 
                                    brush_color='#FFFFFF', mask_opacity=0.5)
                                    
                ref_mask = gr.Image(label="Reference Mask", source="upload", type="pil")

            with gr.Row():
                ref_mask_option = gr.Radio(["Draw Mask", "Upload with Mask"], label="Reference Mask Input Option", value="Upload with Mask")

            baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", height=512, columns=1)
            with gr.Accordion("Advanced Option", open=True):
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=666)
                gr.Markdown("### Guidelines")
                gr.Markdown(" Users can try using different seeds. For example, seeds like 42 and 123456 may produce different effects.")

    run_local_button = gr.Button(value="Run")

    # #### example #####
    num_examples = len(image_list)
    for i in range(num_examples):
        with gr.Row():
            if i == 0:
                gr.Examples([image_list[i]], inputs=[base_image], label="Examples - Background Image", examples_per_page=1)
                gr.Examples([image_mask_list[i]], inputs=[base_mask], label="Examples - Background Mask", examples_per_page=1)
                gr.Examples([ref_list[i]], inputs=[ref_image], label="Examples - Reference Object", examples_per_page=1)
                gr.Examples([ref_mask_list[i]], inputs=[ref_mask], label="Examples - Reference Mask", examples_per_page=1)
            else:
                gr.Examples([image_list[i]], inputs=[base_image], examples_per_page=1, label="")
                gr.Examples([image_mask_list[i]], inputs=[base_mask], examples_per_page=1, label="")
                gr.Examples([ref_list[i]], inputs=[ref_image], examples_per_page=1, label="")
                gr.Examples([ref_mask_list[i]], inputs=[ref_mask], examples_per_page=1, label="")
        if i < num_examples - 1:
            gr.HTML("<hr>")
    # #### example #####

    run_local_button.click(fn=run_local,
                            inputs=[base_image, base_mask, ref_image, ref_mask, seed, base_mask_option, ref_mask_option],
                            outputs=[baseline_gallery]
                            )
demo.launch()