"""Ways of measuring how far apart two images are."""

import numpy as np


def compare(a, b):
    """The largest difference between two uint8 images, and the % of values within 1.

    Only meaningful when the two were produced by the same operation, which is
    true of blur, denoise and the two point operations (brightness, channels).
    Sharpen, edges and compress are compared against a different method, so
    their pages show no such numbers.
    """
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "max": int(d.max()),
        "within1": "{:.1f}".format(100.0 * float((d <= 1).mean())),
    }


def psnr(a, b):
    """Peak signal to noise ratio, in dB, for two float images in 0..1.

    Higher is closer. Returns None for identical images, where it is infinite.
    """
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return None
    return "{:.1f}".format(10.0 * np.log10(1.0 / mse))


def pixel_loss(a, b):
    """Average error of a channel value, as a percentage of the full 0..255 range.

    The "% loss" in the compression stats, with "% quality" as 100 minus it.
    It reads gently -- a few percent can already be visible ringing -- which is
    why PSNR is shown beside it.
    """
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return 100.0 * float(np.mean(diff)) / 255.0
