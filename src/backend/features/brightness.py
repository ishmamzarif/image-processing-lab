"""Brightness: multiply every value by a gain. A point operation, written as a loop.

How it works
    out = gain * in, for every pixel of every channel, then clipped to 0..255.
    The same gain on all three channels keeps their ratios, so the colours
    keep their hue and only get lighter or darker.

Compared against
    Pillow's ImageEnhance.Brightness, which is the same operation, so the two
    are compared number for number.

Mirrored in the browser
    frontend/static/js/adjust.js repeats this multiply on every slider movement
    for the live preview, so a change here has to be ported there.
"""

import numpy as np
from flask import render_template
from PIL import ImageEnhance

from backend.common.limits import MAX_DIM
from backend.common.metrics import compare
from backend.common.timing import timed, timing_fields
from backend.common.uploads import clamp, form_number, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def brighten(channel, gain):
    """Brightness as a point operation, written out by hand.

    Every output pixel depends on its own input pixel and nothing else, so
    unlike the blur there is no neighbourhood, no padding and no kernel:

        out[y, x] = gain * in[y, x]

    gain 1 leaves the image alone, 0 is black and 2 doubles every value, with
    anything past 255 clipped by the caller. It is a gain rather than an
    offset because that is what Pillow means by brightness (ImageEnhance
    blends the image with black), so the library version is the same
    operation and can be compared number for number.

    adjust.js repeats this multiply in the browser on every movement of the
    slider, so a change here has to be ported there.
    """
    h, w = channel.shape
    out = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            out[y, x] = channel[y, x] * gain

    return out


def brighten_image(source, gain):
    brightened = np.zeros_like(source)
    for c in range(3):  # the same gain on every channel, so the hue is kept
        brightened[:, :, c] = brighten(source[:, :, c], gain)
    return brightened


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_brightness(img, gain):
    """Pillow's brightness, which is the same operation as brighten().

    ImageEnhance.Brightness blends with an all-black image,
    black * (1 - g) + img * g, and with black being zero that is g * img. It
    truncates as well. The one difference is precision: Pillow's C code holds
    the factor as a 32-bit float, so 0.7 is really 0.69999999 there, and a
    product that should land exactly on a whole number (10 * 0.7) comes out a
    hair under it and truncates one level lower. Swept over every value at
    every slider step, that accounts for every difference.
    """
    return (
        np.asarray(ImageEnhance.Brightness(img).enhance(gain), dtype=np.uint8),
        "ImageEnhance.Brightness(img).enhance({:g})".format(gain),
    )


# ---------------------------------------------------------------------------
# Web route: POST /brightness
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    # the same range as the slider: at 2x everything brighter than mid-grey is
    # already clipped white, so further up there is little image left
    gain = clamp(form_number("gain", 1.0), 0.0, 2.0)

    # one multiply per value, so this would cope with far more than MAX_DIM,
    # but the live preview in the browser reproduces this exact cap
    img, original = open_image(file, MAX_DIM)
    source = original.astype(np.float64)

    brightened, elapsed = timed(brighten_image, source, gain)

    # the share pushed past 255, which the clip flattens for good
    clipped = 100.0 * float((brightened > 255.0).mean())
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)

    (library, lib_call), elapsed_lib = timed(library_brightness, img, gain)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        brightened=to_data_uri(brightened),
        library=to_data_uri(library),
        lib_call=lib_call,
        lib_name="Pillow",
        stats=compare(brightened, library),   # same operation, so the numbers mean something
        gain="{:g}".format(gain),
        clipped="{:.1f}".format(clipped) if clipped > 0 else None,
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
