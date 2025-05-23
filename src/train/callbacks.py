import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os
from ..models.pipeline_tools import encode_images, prepare_text_input
import json
import math

try:
    import wandb
except ImportError:
    wandb = None

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)



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

def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 2 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
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
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if (self.total_steps % self.save_interval == 0 or self.total_steps == 1) and self.total_steps < 15500:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals   or self.total_steps == 1
        if self.total_steps % self.sample_interval == 0 or self.total_steps == 1:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                self.total_steps,
                f"{self.save_path}/{self.run_name}/output_diptych",
                f"lora_{self.total_steps}",
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        steps,
        save_path,
        file_name,
    ):
        # TODO: change this two variables to parameters

        seed = 42
        size = (768, 768)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        def final_inference(test_dir):
            
            os.makedirs(os.path.join(save_path, f"{file_name}_seed{seed}"), exist_ok=True)

            source_dir = f"{test_dir}/tar_image"
            mask_dir = f"{test_dir}/tar_mask"
            reference_dir = f"{test_dir}/ref_image"
            ref_mask_dir = f"{test_dir}/ref_mask"

            source_images = [f for f in os.listdir(source_dir) if f.endswith(".png") or f.endswith(".jpg")]

            for source_image_filename in source_images:

                source_image_path = os.path.join(source_dir, source_image_filename)
                mask_image_path = os.path.join(mask_dir, source_image_filename)
  
                if os.path.exists(mask_image_path):

                    print(f"Processing {source_image_filename}...")

                    ref_image_path = os.path.join(reference_dir, source_image_filename)
                    ref_mask_path = os.path.join(ref_mask_dir, source_image_filename)

                    ref_image = cv2.imread(ref_image_path)
                    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                    tar_image = cv2.imread(source_image_path)
                    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
                    ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]
                    tar_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]

                    if tar_mask.shape != tar_image.shape:
                        tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

                    ref_box_yyxx = get_bbox_from_mask(ref_mask)
                    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
                    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
                    y1,y2,x1,x2 = ref_box_yyxx
                    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:] 
                    ref_mask = ref_mask[y1:y2,x1:x2] 
                    ratio = 1.3
                    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 

                    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 

                    # kernel = np.ones((7, 7), np.uint8)
                    # iterations = 2
                    # tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

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
                    pipe_prior_output = pl_module.flux_redux(Image.fromarray(masked_ref_image))

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

                    generator = torch.Generator(pl_module.device).manual_seed(seed)
                    edited_image = pl_module.flux_fill_pipe(
                        image=diptych_ref_tar,
                        mask_image=mask_diptych,
                        height=mask_diptych.size[1],
                        width=mask_diptych.size[0],
                        max_sequence_length=512,
                        generator=generator,
                        **pipe_prior_output,  # Use the output from the prior redux model
                    ).images[0]

                    t_width, t_height = edited_image.size
                    start_x = t_width // 2
                    edited_image = edited_image.crop((start_x, 0, t_width, t_height))

                    edited_image = np.array(edited_image)
                    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
                    edited_image = Image.fromarray(edited_image)

                    # Save the result
                    edited_image_save_path = os.path.join(save_path, f"{file_name}_seed{seed}", f"{source_image_filename}")
                    edited_image.save(edited_image_save_path)
                else:
                    print(f"No mask for {source_image_filename}, skipping.")
                    
        # replace test_dir with the path to your test directory, like data/test/garment
        test_dir = "path/to/test"
        
        if os.path.exists(test_dir):
            final_inference(test_dir)

        

       
           