import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset import PairedImageDataset, tensor_to_pil
from src.models import HDRNet
from src.checkpoint import load_checkpoint
from src.augmentations import apply_basic_transform, apply_advanced_transform
from src.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run Screen2Photo inference.")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to inference dataset root with cover/ and final/ folders.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        help="Directory for generated images.",
    )

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--advanced_transform",
        action="store_true",
        help="Apply advanced synthetic camera transform after generator.",
    )

    parser.add_argument(
        "--no_transform",
        action="store_true",
        help="Do not apply any synthetic transform after generator.",
    )

    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save side-by-side cover/generated/target visualizations.",
    )

    return parser.parse_args()


def save_triplet_visualization(
    cover_image: Image.Image,
    generated_image: Image.Image,
    target_image: Image.Image,
    output_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cover_image)
    axes[0].set_title("Cover Input")
    axes[0].axis("off")

    axes[1].imshow(generated_image)
    axes[1].set_title("Generated")
    axes[1].axis("off")

    axes[2].imshow(target_image)
    axes[2].set_title("Target")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generated_dir = os.path.join(args.output_dir, "generated")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")

    os.makedirs(generated_dir, exist_ok=True)

    if args.save_visualizations:
        os.makedirs(visualizations_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PairedImageDataset(
        args.dataset_root,
        image_size=args.image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Inference pairs: {len(dataset)}")

    generator = HDRNet(input_nc=3, output_nc=3).to(device)

    load_checkpoint(
        args.checkpoint,
        generator=generator,
        map_location=device,
    )

    generator.eval()

    mse_list = []
    psnr_list = []
    ssim_list = []

    if args.no_transform:
        transform_func = None
    elif args.advanced_transform:
        transform_func = apply_advanced_transform
    else:
        transform_func = apply_basic_transform

    image_index = 0

    for batch in tqdm(dataloader, desc="Inference"):
        cover_batch = batch["cover"].to(device)
        final_batch = batch["final"].to(device)

        with torch.no_grad():
            fake_batch = generator(cover_batch)

        batch_size = cover_batch.size(0)

        for item_idx in range(batch_size):
            cover_tensor = cover_batch[item_idx].cpu()
            final_tensor = final_batch[item_idx].cpu()
            fake_tensor = fake_batch[item_idx].cpu()

            cover_image = tensor_to_pil(cover_tensor)
            final_image = tensor_to_pil(final_tensor)
            fake_image = tensor_to_pil(fake_tensor)

            if transform_func is not None:
                generated_image = transform_func(
                    fake_image,
                    max_angle=3,
                    max_pad=0.01,
                    distortion_scale=0.05,
                    noise_std=0.1,
                )
            else:
                generated_image = fake_image

            target_resized = final_image.resize(
                generated_image.size,
                Image.Resampling.BILINEAR,
            )

            mse_val, psnr_val, ssim_val = compute_metrics(
                target_resized,
                generated_image,
            )

            mse_list.append(mse_val)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            generated_path = os.path.join(
                generated_dir,
                f"generated_{image_index:05d}.png",
            )

            generated_image.save(generated_path)

            if args.save_visualizations:
                visualization_path = os.path.join(
                    visualizations_dir,
                    f"comparison_{image_index:05d}.png",
                )

                save_triplet_visualization(
                    cover_image=cover_image,
                    generated_image=generated_image,
                    target_image=final_image,
                    output_path=visualization_path,
                )

            image_index += 1

    if len(mse_list) == 0:
        print("No images processed.")
        return

    avg_mse = sum(mse_list) / len(mse_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"Average MSE : {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    metrics_path = os.path.join(args.output_dir, "metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"Average MSE : {avg_mse:.4f}\n")
        file.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        file.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Generated images saved to: {generated_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()