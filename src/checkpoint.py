import os
import torch


def load_checkpoint(
    checkpoint_path,
    generator,
    discriminator=None,
    stn_module=None,
    optimizer_G=None,
    optimizer_D=None,
    map_location="cpu",
):
    """
    Loads checkpoint.

    Can be used both for training resume and generator-only inference.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    else:
        generator.load_state_dict(checkpoint)

    if discriminator is not None and "discriminator" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator"])

    if stn_module is not None and "stn_module" in checkpoint:
        stn_module.load_state_dict(checkpoint["stn_module"])

    if optimizer_G is not None and "optimizer_G" in checkpoint:
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])

    if optimizer_D is not None and "optimizer_D" in checkpoint:
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])

    start_epoch = checkpoint.get("epoch", 0)

    print(f"Checkpoint loaded. Start epoch: {start_epoch}")

    return start_epoch


def save_checkpoint(
    checkpoint_path,
    epoch,
    generator,
    discriminator,
    stn_module,
    optimizer_G,
    optimizer_D,
):
    """
    Saves full training checkpoint.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "stn_module": stn_module.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")