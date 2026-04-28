import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset import PairedImageDataset
from src.models import HDRNet, STN, MultiScaleDiscriminator
from src.checkpoint import load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Screen2Photo model.")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root with cover/ and final/ folders.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory for checkpoints.",
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resume.",
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--lambda_l1", type=float, default=2.0)
    parser.add_argument("--lambda_feat", type=float, default=10.0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_interval", type=int, default=10)

    parser.add_argument(
        "--freeze_stn_epoch",
        type=int,
        default=200,
        help="Epoch after which STN parameters are frozen.",
    )

    return parser.parse_args()


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def compute_generator_losses(
    discriminator,
    criterion_gan,
    criterion_l1,
    fake_ab,
    real_ab,
    fake_b,
    fixed_real_b,
    lambda_l1,
    lambda_feat,
):
    pred_fake = discriminator(fake_ab)

    with torch.no_grad():
        pred_real_for_fm = discriminator(real_ab)

    loss_gan = 0.0

    for scale_out in pred_fake:
        logits = scale_out[-1]
        valid = torch.ones_like(logits, device=logits.device)
        loss_gan += criterion_gan(logits, valid)

    loss_gan = loss_gan / discriminator.num_D

    loss_fm = 0.0
    total_fm_terms = 0

    for scale_idx in range(discriminator.num_D):
        num_intermediate = len(pred_fake[scale_idx]) - 1

        for layer_idx in range(num_intermediate):
            loss_fm += criterion_l1(
                pred_fake[scale_idx][layer_idx],
                pred_real_for_fm[scale_idx][layer_idx],
            )
            total_fm_terms += 1

    if total_fm_terms > 0:
        loss_fm = loss_fm / total_fm_terms

    loss_l1 = criterion_l1(fake_b, fixed_real_b)

    loss_g = loss_gan + lambda_l1 * loss_l1 + lambda_feat * loss_fm

    return loss_g, loss_gan, loss_l1, loss_fm


def compute_discriminator_loss(
    discriminator,
    criterion_gan,
    real_ab,
    fake_ab,
):
    """
    Critical fix compared to the original notebook:
    pred_real is computed without torch.no_grad(), otherwise the discriminator
    does not get gradients from real samples.
    """
    pred_real = discriminator(real_ab)
    pred_fake = discriminator(fake_ab.detach())

    loss_d = 0.0

    for scale_idx in range(discriminator.num_D):
        real_logits = pred_real[scale_idx][-1]
        fake_logits = pred_fake[scale_idx][-1]

        valid = torch.ones_like(real_logits, device=real_logits.device)
        fake = torch.zeros_like(fake_logits, device=fake_logits.device)

        loss_real = criterion_gan(real_logits, valid)
        loss_fake = criterion_gan(fake_logits, fake)

        loss_d += 0.5 * (loss_real + loss_fake)

    loss_d = loss_d / discriminator.num_D

    return loss_d


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = PairedImageDataset(
        args.dataset_root,
        image_size=args.image_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training pairs: {len(train_dataset)}")

    generator = HDRNet(input_nc=3, output_nc=3).to(device)
    stn_module = STN().to(device)

    discriminator = MultiScaleDiscriminator(
        input_nc=6,
        ndf=64,
        n_layers=3,
        num_D=3,
    ).to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    optimizer_g = optim.Adam(
        list(generator.parameters()) + list(stn_module.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    start_epoch = 0

    if args.resume_checkpoint is not None:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(args.resume_checkpoint)

        start_epoch = load_checkpoint(
            args.resume_checkpoint,
            generator=generator,
            discriminator=discriminator,
            stn_module=stn_module,
            optimizer_G=optimizer_g,
            optimizer_D=optimizer_d,
            map_location=device,
        )

    if start_epoch >= args.freeze_stn_epoch:
        set_requires_grad(stn_module, False)
        print(f"STN is frozen from epoch {args.freeze_stn_epoch}")

    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()

        if epoch >= args.freeze_stn_epoch:
            stn_module.eval()
            set_requires_grad(stn_module, False)
        else:
            stn_module.train()
            set_requires_grad(stn_module, True)

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        last_loss_g = None
        last_loss_d = None
        last_loss_l1 = None
        last_loss_fm = None

        for batch in progress_bar:
            real_a = batch["cover"].to(device)
            real_b = batch["final"].to(device)

            fixed_real_b, _ = stn_module(real_b)
            fake_b = generator(real_a)

            real_ab = torch.cat([real_a, fixed_real_b], dim=1)
            fake_ab = torch.cat([real_a, fake_b], dim=1)

            optimizer_g.zero_grad(set_to_none=True)

            loss_g, loss_gan, loss_l1, loss_fm = compute_generator_losses(
                discriminator=discriminator,
                criterion_gan=criterion_gan,
                criterion_l1=criterion_l1,
                fake_ab=fake_ab,
                real_ab=real_ab,
                fake_b=fake_b,
                fixed_real_b=fixed_real_b,
                lambda_l1=args.lambda_l1,
                lambda_feat=args.lambda_feat,
            )

            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad(set_to_none=True)

            loss_d = compute_discriminator_loss(
                discriminator=discriminator,
                criterion_gan=criterion_gan,
                real_ab=real_ab.detach(),
                fake_ab=fake_ab.detach(),
            )

            loss_d.backward()
            optimizer_d.step()

            last_loss_g = loss_g.item()
            last_loss_d = loss_d.item()
            last_loss_l1 = loss_l1.item()
            last_loss_fm = loss_fm.item()

            progress_bar.set_postfix(
                {
                    "loss_G": f"{last_loss_g:.4f}",
                    "loss_D": f"{last_loss_d:.4f}",
                    "L1": f"{last_loss_l1:.4f}",
                    "FM": f"{last_loss_fm:.4f}",
                }
            )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss_G={last_loss_g:.4f}, "
            f"loss_D={last_loss_d:.4f}, "
            f"L1={last_loss_l1:.4f}, "
            f"FM={last_loss_fm:.4f}"
        )

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint_epoch_{epoch + 1}.pth",
            )

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch + 1,
                generator=generator,
                discriminator=discriminator,
                stn_module=stn_module,
                optimizer_G=optimizer_g,
                optimizer_D=optimizer_d,
            )

    final_checkpoint_path = os.path.join(
        args.output_dir,
        f"checkpoint_epoch_{args.epochs}.pth",
    )

    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        epoch=args.epochs,
        generator=generator,
        discriminator=discriminator,
        stn_module=stn_module,
        optimizer_G=optimizer_g,
        optimizer_D=optimizer_d,
    )

    print(f"Training finished. Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()