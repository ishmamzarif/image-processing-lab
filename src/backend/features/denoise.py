"""Denoise: add noise to the image, then remove it in the frequency domain.

The idea the whole thing rests on: noise that is *periodic* in the image is
*localised* in the spectrum. A sinusoid across the picture is a handful of
bright points, far from the centre, and nothing else. Cut those points out and
the stripes vanish while the picture itself, which lives in the low and mid
frequencies, is left alone. Grain has no such structure -- it is spread evenly
over every frequency -- so the only way to reach it is to throw away a whole
band, which takes real detail with it. The filters here show that contrast.

How it works
    1. add_noise      stripes, grain or both are stirred into the clean image
    2. fft2 of luma   one spectrum to look at and to hunt for peaks in
    3. a mask         1 = keep this frequency, 0 = remove it:
                        notch     a hole at each bright peak found
                        gaussian  a soft low-pass circle
                        ideal     a hard low-pass circle (rings)
    4. filter_channels  every colour channel: fft2, multiply by mask, ifft2

    The 2D FFT comes from common/fourier_2d.py, built on transforms.py.

Compared against
    The same mask applied with numpy.fft, which checks the hand-written
    transform number for number. The page also shows PSNR before and after.
"""

import numpy as np
from flask import render_template, request

from backend.common.fourier_2d import fft2, ifft2, luma, pad_to_pow2, shift, unshift
from backend.common.limits import WORK_DIM
from backend.common.metrics import compare, psnr
from backend.common.spectrum import spectrum_plate, spectrum_surface
from backend.common.timing import timed, timing_fields
from backend.common.uploads import form_number, open_image, size_text, to_data_uri, uploaded_file


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


# ---------------------------------------------------------------------------
# Making the noise
# ---------------------------------------------------------------------------

def add_noise(img, kind, amount, seed=0):
    """Corrupt a float image in 0..1. Returns the noisy image, same range.

    kind is "periodic" (stripes), "grain" (random speckle) or "both". The seed
    is fixed so the same image and settings always give the same noise.
    """
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


# ---------------------------------------------------------------------------
# Masks: 1 keeps a frequency, 0 removes it. All are for a centred spectrum.
# ---------------------------------------------------------------------------

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

    Returns the mask and the (row, column) of every peak found.

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

    # repeatedly take the brightest remaining bin, until nothing stands out
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


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def filter_channels(rgb, mask):
    """Transform, multiply by the mask, transform back. Per colour channel."""
    padded, (h, w) = pad_to_pow2(rgb)
    out = np.zeros_like(padded)
    for c in range(3):
        spec = shift(fft2(padded[:, :, c]))
        out[:, :, c] = np.real(ifft2(unshift(spec * mask)))
    return np.clip(out[:h, :w], 0.0, 1.0)


def build_mask(mag, which, cutoff, width, softness):
    """The mask the form asked for. Returns it with the peaks found (notch only)."""
    peaks = []
    if which == "notch":
        mask, peaks = mask_notch(mag, max(8.0, width * 2), width, softness)
    elif which == "gaussian":
        mask = mask_gaussian(mag.shape[0], mag.shape[1], cutoff, softness)
    else:
        mask = mask_ideal(mag.shape[0], mag.shape[1], cutoff)
    return mask, peaks


def denoise_image(noisy, which, cutoff, width, softness):
    """The whole hand-written pipeline. Returns (cleaned, luma spectrum, mask, peaks)."""
    # the spectrum is taken of the luminance: one picture to look at, and one
    # place to hunt for peaks, while the mask itself is applied to all three
    # colour channels
    padded, _ = pad_to_pow2(noisy)
    mag = np.abs(shift(fft2(luma(padded))))

    mask, peaks = build_mask(mag, which, cutoff, width, softness)
    cleaned = filter_channels(noisy, mask)
    return cleaned, mag, mask, peaks


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def numpy_filter(rgb, mask):
    """The same mask applied with numpy's FFT instead of ours.

    NOTE: numpy.fft appears twice in the project, here and in encrypt.py,
    both times in a library baseline rather than in transforms.py, and both
    times purely as a check on the hand-written transform. Delete this function
    and the `library=` argument below if the assignment forbids the import
    anywhere in the tree.
    """
    padded, (h, w) = pad_to_pow2(rgb)
    out = np.zeros_like(padded)
    for c in range(3):
        spec = np.fft.fftshift(np.fft.fft2(padded[:, :, c]))
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask)))
    return np.clip(out[:h, :w], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Web route: POST /denoise
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    kind = request.form.get("noise", "periodic")
    amount = form_number("noise_amount", 0.3)
    which = request.form.get("filter", "notch")
    cutoff = form_number("fcut", 45)
    width = form_number("nwidth", 3)
    softness = form_number("softness", 20)
    view_mode = request.form.get("specview", "mask")

    _, original = open_image(file, WORK_DIM)   # numpy_filter needs no Pillow image
    clean = original.astype(np.float64) / 255.0

    noisy = add_noise(clean, kind, amount, seed=0)

    (cleaned, mag, mask, peaks), elapsed = timed(denoise_image, noisy, which, cutoff, width, softness)

    library, elapsed_lib = timed(numpy_filter, noisy, mask)

    cleaned_u8 = (cleaned * 255).astype(np.uint8)
    library_u8 = (library * 255).astype(np.uint8)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        noisy=to_data_uri((noisy * 255).astype(np.uint8)),
        spectrum=to_data_uri(spectrum_plate(mag, mask, view_mode)),
        surface=spectrum_surface(mag, mask),   # the same spectrum in 3D (spectrum-3d.js)
        denoised=to_data_uri(cleaned_u8),
        library=to_data_uri(library_u8),
        lib_call="numpy.fft.fft2 / ifft2",
        stats=compare(cleaned_u8, library_u8),
        noise=kind,
        noise_amount="{:g}".format(amount),
        filt=which,
        fcut="{:g}".format(cutoff),
        nwidth="{:g}".format(width),
        softness="{:g}".format(softness),
        specview=view_mode,
        peaks=len(peaks),
        kept="{:.1f}".format(100.0 * float(np.mean(mask))),
        psnr_noisy=psnr(clean, noisy),
        psnr_clean=psnr(clean, cleaned),
        size=size_text(original),
        spec_size=size_text(mag),
        **timing_fields(elapsed, elapsed_lib),
    )
