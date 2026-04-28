import numpy as np
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def compute_metrics(img_target: Image.Image, img_generated: Image.Image):
    """
    Computes MSE, PSNR and SSIM between two PIL images of the same size.
    """
    np_target = np.array(img_target).astype(np.float32) / 255.0
    np_generated = np.array(img_generated).astype(np.float32) / 255.0

    mse_val = mean_squared_error(np_target, np_generated)

    psnr_val = peak_signal_noise_ratio(
        np_target,
        np_generated,
        data_range=1.0,
    )

    h, w, _ = np_target.shape

    max_win_size = min(h, w, 7)

    if max_win_size % 2 == 0:
        max_win_size -= 1

    max_win_size = max(max_win_size, 3)

    ssim_val = structural_similarity(
        np_target,
        np_generated,
        data_range=1.0,
        channel_axis=-1,
        win_size=max_win_size,
    )

    return mse_val, psnr_val, ssim_val