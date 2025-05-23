import lightning as L
from diffusers.pipelines import FluxFillPipeline, FluxPriorReduxPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt
from PIL import Image
from .transformer import tranformer_forward
from .pipeline_tools import encode_images, prepare_text_input, Flux_fill_encode_masks_images
from .image_project import image_output

class InsertAnything(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        flux_redux_id: str, 
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe: FluxFillPipeline = (
            FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)
        )

        self.flux_redux: FluxPriorReduxPipeline = (
            FluxPriorReduxPipeline.from_pretrained(flux_redux_id).to(dtype=dtype).to(device)
        )

        self.flux_redux.image_embedder.requires_grad_(False).eval()
        self.flux_redux.image_encoder.requires_grad_(False).eval()


        self.transformer = self.flux_fill_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_fill_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_fill_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_fill_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        

        imgs = batch["result"]

        src = batch["src"]
        mask = batch["mask"]

        ref = batch["ref"]

        prompt_embeds = []
        pooled_prompt_embeds = []

        for i in range(ref.shape[0]):

            image_tensor = ref[i].cpu()

            image_tensor = image_tensor.permute(1, 2, 0)

            image_numpy = image_tensor.numpy()

            pil_image = Image.fromarray((image_numpy * 255).astype('uint8'))

            prompt_embed, pooled_prompt_embed = image_output(self.flux_redux, pil_image, self.device)


            prompt_embeds.append(prompt_embed.squeeze(1))
        
            pooled_prompt_embeds.append(pooled_prompt_embed.squeeze(1))     


        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)

        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_fill_pipe, prompt_embeds=prompt_embeds.to(self.device), pooled_prompt_embeds=pooled_prompt_embeds.to(self.device)
        )

        # Prepare inputs
        with torch.no_grad():


            # Prepare image input
            x_0, img_ids = encode_images(self.flux_fill_pipe, imgs)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare conditions
            src_latents, mask_latents = Flux_fill_encode_masks_images(self.flux_fill_pipe, src, mask)

            condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            hidden_states=torch.cat((x_t, condition_latents), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss






