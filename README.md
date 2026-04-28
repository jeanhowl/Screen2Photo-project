# Screen2Photo

Screen2Photo is an image-to-image translation project that converts clean screen/cover images into photo-like screen images.

The model uses:
- HDRNet-style generator
- Spatial Transformer Network for target alignment
- Multi-scale PatchGAN discriminator
- L1 reconstruction loss
- adversarial loss
- feature matching loss

## Dataset format

Expected structure:

```text
dataset_root/
  cover/
    folder_1/
      image_1.png
  final/
    folder_1/
      image_1.jpg