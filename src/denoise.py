"""Frequency-domain denoising, built on the transforms in transforms.py.

The idea the whole thing rests on: noise that is *periodic* in the image is
*localised* in the spectrum. A sinusoid across the picture is a handful of
bright points, far from the centre, and nothing else. Cut those points out and
the stripes vanish while the picture itself, which lives in the low and mid
frequencies, is left alone. Grain has no such structure -- it is spread evenly
over every frequency -- so the only way to reach it is to throw away a whole
band, which takes real detail with it. The two filters here show that contrast.

Everything transform-related comes from transforms.py. numpy is used for array
arithmetic only, exactly as that module requires.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transforms import FFTTransformer, next_power_of_two  # noqa: E402

ENGINE = FFTTransformer()

# Worked at this size. The transform is a Python-level butterfly loop, so cost
# is real: 256x256 is about 1.5 s for three channels, 512x512 about eight times
# that. The spectrum is also easier to read when it is not enormous.
WORK_DIM = 256

# how many peaks the notch filter will chase before giving up
MAX_NOTCHES = 12

# A peak has to stand this many times above the 99.9th percentile of the rest of
# the spectrum before it is treated as noise. Measured on real spectra, a
# genuine sinusoid lands around 13x while the brightest bin of pure grain
# reaches only about 1.8x, so anything in the middle separates the two cleanly.
PEAK_FACTOR = 4.0

# the sinusoids stirred in as "periodic" noise, as (cycles down, cycles across).
# Chosen to sit well away from the centre of the spectrum, and away from each
# other, so each shows up as its own isolated pair of points.
PATTERNS = ((0, 27), (19, 19), (23, -11))


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
    return np.roll(a, (-(a.shape[0] // 2), -(a.shape[1] // 2)), axis=(0, 1))


def pad_to_pow2(a):
    """Reflect the image out to power-of-two sides, which the radix-2 FFT needs.

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
    """ITU-R 601-2, the same weights PIL's convert("L") uses."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def add_noise(img, kind, amount, seed=0):
    """Corrupt a float image in 0..1. Returns the noisy image, same range."""
    out = img.copy()
    h, w = img.shape[:2]

    if kind in ("periodic", "both"):
        yy, xx = np.mgrid[0:h, 0:w]
        amp = amount * 0.5 / len(PATTERNS)
        for fy, fx in PATTERNS:
            wave = np.sin(2 * np.pi * (fx * xx / w + fy * yy / h))
            out = out + amp * wave[..., None]

    if kind in ("grain", "both"):
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, amount * 0.35, img.shape)

    return np.clip(out, 0.0, 1.0)


def radius(h, w):
    """Distance of each bin from the centre of a shifted spectrum."""
    v = np.arange(h)[:, None] - h // 2
    u = np.arange(w)[None, :] - w // 2
    return np.sqrt(v ** 2 + u ** 2)


def mask_ideal(h, w, cutoff):
    """Keep everything inside a circle, drop the rest. Sharp, and it rings."""
    return (radius(h, w) <= cutoff).astype(np.float64)


def mask_gaussian(h, w, cutoff, softness):
    """Fade out with distance instead of cutting. No ringing, more blur."""
    sigma = max(1e-6, cutoff * (1.0 + softness / 100.0))
    return np.exp(-(radius(h, w) ** 2) / (2.0 * sigma ** 2))


def mask_notch(mag, protect, width, softness):
    """Punch a hole at each isolated bright point, and leave everything else.

    The peaks are found in the spectrum rather than assumed from the noise that
    was added, so this works the same way on an image that arrived already
    striped.
    """
    h, w = mag.shape
    r = radius(h, w)

    # the centre is the picture itself, and it is always the brightest thing
    # there is, so it is taken off the table before looking for peaks
    hunting = mag.copy()
    hunting[r <= protect] = 0.0

    # A quantile, not a mean: the mean is dragged around by however much
    # ordinary picture there is out here, while the 99.9th percentile tracks the
    # brightest *background* bin, which is exactly what a peak has to beat.
    outside = hunting[r > protect]
    floor = float(np.quantile(outside, 0.999)) * PEAK_FACTOR if outside.size else 0.0

    vv = np.arange(h)[:, None]
    uu = np.arange(w)[None, :]
    mask = np.ones((h, w), dtype=np.float64)
    found = []

    for _ in range(MAX_NOTCHES):
        peak = float(hunting.max())
        if peak <= floor:
            break
        py, px = np.unravel_index(int(np.argmax(hunting)), (h, w))
        found.append((int(py), int(px)))

        d = np.sqrt((vv - py) ** 2 + (uu - px) ** 2)
        if softness <= 0:
            mask *= (d > width)
        else:
            sigma = max(0.35, width * (softness / 100.0) + width * 0.35)
            mask *= 1.0 - np.exp(-(d ** 2) / (2.0 * sigma ** 2))

        # blank this peak's neighbourhood so the next pass finds a different one
        hunting[d <= max(width * 2.0, 3.0)] = 0.0

    return mask, found


def spectrum_picture(mag):
    """log(1 + |X|), scaled to 0..1.

    Without the log this is a single white dot on black: the centre bin is
    orders of magnitude above everything else.
    """
    s = np.log1p(mag)
    top = s.max()
    return s / top if top > 0 else s


def filter_channels(rgb, mask):
    """Transform, multiply by the mask, transform back. Per colour channel."""
    padded, (h, w) = pad_to_pow2(rgb)
    out = np.zeros_like(padded)
    for c in range(3):
        spec = shift(fft2(padded[:, :, c]))
        out[:, :, c] = np.real(ifft2(unshift(spec * mask)))
    return np.clip(out[:h, :w], 0.0, 1.0)
