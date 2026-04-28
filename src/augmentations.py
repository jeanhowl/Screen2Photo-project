import random

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def add_gaussian_noise(image: Image.Image, mean: float = 0, std: float = 2) -> Image.Image:
    """
    Adds weak Gaussian noise to a PIL image.
    """
    img_np = np.array(image)
    noise = np.random.normal(mean, std, img_np.shape).astype(np.float32)

    noisy_img = img_np.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img)


def apply_perspective_transform(
    image: Image.Image,
    distortion_scale: float = 0.1,
) -> Image.Image:
    """
    Applies random perspective distortion to a PIL image.
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    src_points = np.float32(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]
    )

    dst_points = src_points + np.random.uniform(
        -distortion_scale,
        distortion_scale,
        size=(4, 2),
    ) * [w, h]

    dst_points = dst_points.astype(np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img_np, matrix, (w, h))

    return Image.fromarray(warped)


def apply_basic_transform(
    image: Image.Image,
    max_angle: float = 10,
    max_pad: float = 0.1,
    noise_std: float = 2,
) -> Image.Image:
    """
    Applies simple synthetic camera-like augmentations:
    rotation, blur, padding, and noise.
    """
    angle = random.uniform(-max_angle, max_angle)

    rotated = image.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(10, 10, 10),
    )

    rotated = rotated.filter(ImageFilter.GaussianBlur(radius=0.5))

    w, h = rotated.size

    pad_left = int(random.uniform(0, max_pad) * w)
    pad_right = int(random.uniform(0, max_pad) * w)
    pad_top = int(random.uniform(0, max_pad) * h)
    pad_bottom = int(random.uniform(0, max_pad) * h)

    padded = ImageOps.expand(
        rotated,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(10, 10, 10),
    )

    if noise_std > 0:
        padded = add_gaussian_noise(padded, std=noise_std)

    return padded


def apply_advanced_transform(
    image: Image.Image,
    max_angle: float = 3,
    max_pad: float = 0.1,
    distortion_scale: float = 0.1,
    noise_std: float = 0.2,
) -> Image.Image:
    """
    Applies stronger synthetic camera-like augmentations:
    perspective distortion, rotation, blur, padding, and noise.
    """
    persp = apply_perspective_transform(image, distortion_scale)

    angle = random.uniform(-max_angle, max_angle)

    rotated = persp.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(10, 10, 10),
    )

    rotated = rotated.filter(ImageFilter.GaussianBlur(radius=0.5))

    w, h = rotated.size

    pad_left = int(random.uniform(0, max_pad) * w)
    pad_right = int(random.uniform(0, max_pad) * w)
    pad_top = int(random.uniform(0, max_pad) * h)
    pad_bottom = int(random.uniform(0, max_pad) * h)

    padded = ImageOps.expand(
        rotated,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(10, 10, 10),
    )

    if noise_std > 0:
        padded = add_gaussian_noise(padded, std=noise_std)

    return padded