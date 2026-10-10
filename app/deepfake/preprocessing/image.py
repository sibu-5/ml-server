"""
Image Feature Extraction — Frequency & Spatial Feature Analysis.

Extracts spectral (FFT/DCT), spatial noise, and chrominance distribution
features from PIL images to assist deepfake detection algorithms.
"""

import numpy as np
from PIL import Image


def extract_image_features(img: Image.Image) -> dict:
    """
    Extract forensic features from a validated PIL Image.

    Returns:
        dict containing spectral_variance, high_freq_ratio,
        color_channel_variance, and spatial_noise_level.
    """
    # Standardize to RGB
    rgb_img = img.convert("RGB")
    arr = np.array(rgb_img, dtype=np.float32)

    height, width, _ = arr.shape

    # 1. Luminance conversion (0.299 R + 0.587 G + 0.114 B)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # 2. 2D Fast Fourier Transform (FFT) for high-frequency spectral artifacts
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)

    cy, cx = height // 2, width // 2
    r = min(height, width) // 4
    y, x = np.ogrid[:height, :width]
    mask_low = (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2

    total_energy = np.sum(magnitude_spectrum) + 1e-8
    low_freq_energy = np.sum(magnitude_spectrum[mask_low])
    high_freq_energy = total_energy - low_freq_energy
    high_freq_ratio = float(high_freq_energy / total_energy)
    spectral_variance = float(np.var(np.log1p(magnitude_spectrum)))

    # 3. Spatial High-pass noise residual (Laplacian kernel approximation)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Simple valid boundary convolution for noise estimation
    h_idx, w_idx = min(height, 256), min(width, 256)
    sub_gray = gray[:h_idx, :w_idx]

    # Manual 3x3 convolution for spatial noise estimation
    padded = np.pad(sub_gray, 1, mode="edge")
    conv = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    spatial_noise_level = float(np.std(conv))

    # 4. Color channel variance
    r_var = float(np.var(arr[:, :, 0]))
    g_var = float(np.var(arr[:, :, 1]))
    b_var = float(np.var(arr[:, :, 2]))
    color_channel_variance = float(np.mean([r_var, g_var, b_var]))

    return {
        "spectral_variance": round(spectral_variance, 4),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "spatial_noise_level": round(spatial_noise_level, 4),
        "color_channel_variance": round(color_channel_variance, 4),
        "width": width,
        "height": height,
    }
