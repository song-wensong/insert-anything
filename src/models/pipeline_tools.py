from diffusers.pipelines import FluxPipeline, FluxFillPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor
import torch


def Flux_fill_encode_masks_images(pipeline, images, masks):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)

    batch_size, num_channels_latents, height, width = images.shape

    masks = pipeline.mask_processor.preprocess(masks)
    masks = masks.to(pipeline.device).to(pipeline.dtype)
    masks = masks[:, 0, :, :] 
    masks = masks.view(
        batch_size, height, pipeline.vae_scale_factor, width, pipeline.vae_scale_factor
    )  
    masks = masks.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    masks = masks.reshape(
        batch_size, pipeline.vae_scale_factor * pipeline.vae_scale_factor, height, width
    ) 

    masks_tokens = pipeline._pack_latents(
        masks,
        batch_size,
        pipeline.vae_scale_factor * pipeline.vae_scale_factor,
        height,
        width,
    )


    return images_tokens, masks_tokens




def encode_images(pipeline, images):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline, prompt=None, prompt_embeds=None, pooled_prompt_embeds=None, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def prepare_text_input_catvton(pipeline, prompt, max_sequence_length=512):
    
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids