"""Sharpen: unsharp masking done in the frequency domain.

How it works
    For each colour channel, take its continuous Fourier transform, remove the
    low frequencies (the smooth, slowly changing part of the picture), and
    transform back. What is left is the fine detail. Adding `amount` times that
    detail back onto the image makes edges and texture stand out:

        sharpened = image + amount * high_pass(image)

    The transform machinery lives in common/continuous_ft.py, shared with edges.

Compared against
    Pillow's UnsharpMask, which does the same idea in the spatial domain
    (subtracting a blur). It is a different method, so the page shows the two
    side by side without number-for-number stats.
"""

import numpy as np
from flask import render_template
from PIL import ImageFilter

from backend.common.continuous_ft import high_pass_detail
from backend.common.limits import MAX_DIM
from backend.common.timing import timed, timing_fields
from backend.common.uploads import form_number, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def sharpen_image(source, cutoff, amount):
    """Add `amount` times each channel's high-frequency detail back onto it (float 0..1 in)."""
    sharpened = np.zeros_like(source)
    for c in range(3):
        detail = high_pass_detail(source[:, :, c], cutoff)
        sharpened[:, :, c] = source[:, :, c] + amount * detail
    return sharpened


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_sharpen(img, cutoff, amount):
    """Pillow's sharpen, which is not the same method.

    UnsharpMask subtracts a Gaussian blur in the spatial domain; ours subtracts
    a Gaussian low-pass in the frequency domain. Those are the same operation
    seen from two sides, so the parameters do translate: a Gaussian of spatial
    width r has frequency width N / (2*pi*r).
    """
    # A Gaussian of spatial width r corresponds to a frequency width N/(2*pi*r),
    # so inverting that gives the radius matching our cutoff. Measured against
    # UnsharpMask this lands within one step every time.
    side = float(np.sqrt(img.size[0] * img.size[1]))
    radius = float(np.clip(side / (2.0 * np.pi * max(cutoff, 1.0)), 0.3, 20.0))
    percent = int(np.clip(amount * 100.0, 0, 500))
    return (
        # threshold 0: UnsharpMask normally skips areas whose local contrast is
        # under the threshold, and we have no equivalent, so leaving it on would
        # be comparing against a filter doing something extra
        np.asarray(img.filter(ImageFilter.UnsharpMask(radius, percent, 0)), dtype=np.uint8),
        "ImageFilter.UnsharpMask({:.2f}, {}, 0)".format(radius, percent),
    )


# ---------------------------------------------------------------------------
# Web route: POST /sharpen
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    cutoff = form_number("cutoff", 15)
    amount = form_number("amount", 1.0)

    img, original = open_image(file, MAX_DIM)
    source = original.astype(np.float64) / 255.0

    sharpened, elapsed = timed(sharpen_image, source, cutoff, amount)
    sharpened = np.clip(sharpened * 255.0, 0, 255).astype(np.uint8)

    (library, lib_call), elapsed_lib = timed(library_sharpen, img, cutoff, amount)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        sharpened=to_data_uri(sharpened),
        library=to_data_uri(library),
        lib_call=lib_call,
        different_method=True,            # no numbers: it is not the same algorithm
        cutoff="{:g}".format(cutoff),
        amount="{:g}".format(amount),
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
