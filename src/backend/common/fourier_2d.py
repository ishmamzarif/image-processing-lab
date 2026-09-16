"""The 2D discrete Fourier transform, built out of the 1D FFT in transforms.py.

Denoise and compress both work on an image's spectrum, and this is where they
get it. Nothing here calls a library transform: numpy only does array
arithmetic, as transforms.py requires.

The usual round trip looks like this:

    padded, (h, w) = pad_to_pow2(image)     # the radix-2 FFT needs 2^k sides
    spectrum = shift(fft2(padded))          # zero frequency moved to the middle
    ... change the spectrum ...
    back = ifft2(unshift(spectrum)).real[:h, :w]
"""

import numpy as np

from backend.transforms import FFTTransformer, next_power_of_two

ENGINE = FFTTransformer()


def fft2(a):
    """2D forward transform, done separably: every row, then every column.

        X[u,v] = sum_m sum_n x[m,n] e^(-2pi i (um/M + vn/N))
               = sum_m e^(-2pi i um/M) [ sum_n x[m,n] e^(-2pi i vn/N) ]

    The bracket is a transform of one row, and the outer sum is then a
    transform down each column of that result.
    """
    rows = np.array([ENGINE.transform(r) for r in a])
    return np.array([ENGINE.transform(c) for c in rows.T]).T


def ifft2(spectrum):
    """2D inverse. Each 1D inverse carries its own 1/N, so the 1/(M*N) is done."""
    rows = np.array([ENGINE.inverse(r) for r in spectrum])
    return np.array([ENGINE.inverse(c) for c in rows.T]).T


def shift(a):
    """Move the zero frequency to the middle, so the picture is centred."""
    return np.roll(a, (a.shape[0] // 2, a.shape[1] // 2), axis=(0, 1))


def unshift(a):
    """Undo shift(): zero frequency back to the corner, where ifft2 expects it."""
    return np.roll(a, (-(a.shape[0] // 2), -(a.shape[1] // 2)), axis=(0, 1))


def pad_to_pow2(a):
    """Reflect the image out to power-of-two sides, which the radix-2 FFT needs.

    Returns the padded array and the original (height, width), so the caller
    can crop the padding off again afterwards.

    Reflecting rather than zero-filling matters: a hard edge is a step, and a
    step is broadband, so zero padding would scatter energy across the whole
    spectrum and swamp the very peaks we are trying to find.
    """
    h, w = a.shape[:2]
    ph, pw = next_power_of_two(h), next_power_of_two(w)
    if (ph, pw) == (h, w):
        return a, (h, w)
    pad = [(0, ph - h), (0, pw - w)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, mode="reflect"), (h, w)


def luma(rgb):
    """Brightness of an RGB image: ITU-R 601-2, the same weights PIL's convert("L") uses.

    Both frequency features look at one spectrum, the luma's, to decide what
    to do (which peaks to notch, which coefficients to keep), and then apply
    that decision to all three colour channels.
    """
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
