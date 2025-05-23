from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time
from ..data.all_data import AllDataset
from ..models.my_model import InsertAnything
from .callbacks import TrainingCallback

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)

    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    accessory_train = AllDataset(
        image_dir="data/train/accessory",
    )

    object_train = AllDataset(
        image_dir="data/train/object",
    )

    person_train = AllDataset(
        image_dir="data/train/person",
        data_type="person"
    )

    train_dataset = accessory_train + person_train + object_train

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        # pin_memory=training_config["pin_memory"]
    )

    # Initialize model
    trainable_model = InsertAnything(
        flux_fill_id=config["flux_fill_path"],
        flux_redux_id=config["flux_redux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        # enable_progress_bar=False,
        enable_progress_bar=True,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)

if __name__ == "__main__":
    main()
