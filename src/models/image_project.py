from typing import List, Optional, Union
import torch
from PIL import Image


def image_output(pipeline, image, device):

    if image is not None and isinstance(image, Image.Image):
        batch_size = 1
    elif image is not None and isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]

    device = device

    # 3. Prepare image embeddings
    image_latents = pipeline.encode_image(image, device, 1)

    image_embeds = pipeline.image_embedder(image_latents).image_embeds

    image_embeds = image_embeds.to(device=device)

    # max_sequence_length is 512, t5 encoder hidden size is 4096
    prompt_embeds = torch.zeros((batch_size, 512, 4096), device=device, dtype=image_embeds.dtype)
    # pooled_prompt_embeds is 768, clip text encoder hidden size
    pooled_prompt_embeds = torch.zeros((batch_size, 768), device=device, dtype=image_embeds.dtype)

    # scale & concatenate image and text embeddings
    prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

    prompt_embeds_scale = 1.0
    pooled_prompt_embeds_scale = 1.0

    prompt_embeds_scale = batch_size * [prompt_embeds_scale]
    pooled_prompt_embeds_scale = batch_size * [pooled_prompt_embeds_scale]

    prompt_embeds *= torch.tensor(prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[:, None, None]
    pooled_prompt_embeds *= torch.tensor(pooled_prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[
        :, None
    ]

    # weighted sum
    prompt_embeds = torch.sum(prompt_embeds, dim=0, keepdim=True)
    pooled_prompt_embeds = torch.sum(pooled_prompt_embeds, dim=0, keepdim=True)

    # Offload all models
    # pipeline.maybe_free_model_hooks()

    return (prompt_embeds, pooled_prompt_embeds)
