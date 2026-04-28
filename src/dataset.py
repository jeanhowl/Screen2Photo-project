import os
from typing import Dict

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def tensor_to_pil(tensor_img):
    """
    Converts a normalized tensor in [-1, 1] back to a PIL image in [0, 255].
    """
    unnorm = transforms.Normalize(mean=[-0.5 / 0.5] * 3, std=[1 / 0.5] * 3)
    tensor_img = unnorm(tensor_img).clamp(0, 1)
    pil_img = transforms.ToPILImage()(tensor_img)
    return pil_img


class PairedImageDataset(Dataset):
    """
    Dataset for paired image-to-image translation.

    Expected structure:

    dataset_root/
      cover/
        subfolder/
          image.png
      final/
        subfolder/
          image.jpg or image.png
    """

    def __init__(self, dataset_root: str, image_size: int = 512):
        self.pairs = []

        cover_root = os.path.join(dataset_root, "cover")
        final_root = os.path.join(dataset_root, "final")

        if not os.path.isdir(cover_root):
            raise FileNotFoundError(f"Cover directory not found: {cover_root}")

        if not os.path.isdir(final_root):
            raise FileNotFoundError(f"Final directory not found: {final_root}")

        for subfolder in sorted(os.listdir(cover_root)):
            cover_sub = os.path.join(cover_root, subfolder)
            final_sub = os.path.join(final_root, subfolder)

            if not os.path.isdir(cover_sub) or not os.path.isdir(final_sub):
                continue

            cover_files = sorted(
                [
                    f for f in os.listdir(cover_sub)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )

            for fname in cover_files:
                cover_path = os.path.join(cover_sub, fname)

                base_name = os.path.splitext(fname)[0]
                final_jpg_path = os.path.join(final_sub, base_name + ".jpg")
                final_original_path = os.path.join(final_sub, fname)

                if os.path.exists(final_jpg_path):
                    final_path = final_jpg_path
                elif os.path.exists(final_original_path):
                    final_path = final_original_path
                else:
                    print(f"Warning: no matching final image for {cover_path}")
                    continue

                self.pairs.append((cover_path, final_path))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No image pairs found in dataset: {dataset_root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Dict[str, object]:
        cover_path, final_path = self.pairs[idx]

        cover_img = Image.open(cover_path).convert("RGB")
        final_img = Image.open(final_path).convert("RGB")

        return {
            "cover": self.transform(cover_img),
            "final": self.transform(final_img),
            "cover_path": cover_path,
            "final_path": final_path,
        }