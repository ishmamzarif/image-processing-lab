"""Colour channels: add a separate offset to red, green and blue. A point operation.

How it works
    out = in + offset_c, for every pixel, with a different whole-number offset
    for each channel, then clipped to 0..255. Unequal offsets change the
    balance between the channels, which is how a colour cast is added or
    removed.

Compared against
    Pillow's Image.point with a lookup table, which is the same function, so
    the two are compared number for number (and should match exactly).

Mirrored in the browser
    frontend/static/js/adjust.js repeats this addition for the live preview,
    so a change here has to be ported there.
"""

import numpy as np
from flask import render_template, request

from backend.common.limits import MAX_DIM
from backend.common.metrics import compare
from backend.common.timing import timed, timing_fields
from backend.common.uploads import clamp, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def shift_channel(channel, offset):
    """One colour channel moved up or down by a fixed amount, by hand.

    Another point operation, the same shape of loop as brighten():

        out[y, x] = in[y, x] + offset

    but added rather than multiplied, and with a different offset for each
    channel. That is the difference that matters: a gain shared by all three
    keeps the ratios between them, so the hue survives, while unequal offsets
    change the balance, which is how a colour cast is put in or taken out.
    Whole numbers in, whole numbers out, so the only thing the caller has to
    do is clip at both ends.

    adjust.js repeats this addition in the browser, so a change here has to be
    ported there too.
    """
    h, w = channel.shape
    out = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            out[y, x] = channel[y, x] + offset

    return out


def shift_image(source, offsets):
    shifted = np.zeros_like(source)
    for c in range(3):  # each channel its own offset, which is what moves the hue
        shifted[:, :, c] = shift_channel(source[:, :, c], offsets[c])
    return shifted


def signed(n):
    """An offset as the page shows it: +40, −20 (a true minus sign) or 0."""
    if n > 0:
        return "+{}".format(n)
    if n < 0:
        return "−{}".format(-n)
    return "0"


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_channels(img, offsets):
    """Pillow's version of the channel offsets: Image.point with a lookup table.

    In a point operation the output depends on the input value and nothing
    else, so each channel has only 256 possible answers. Pillow wants them
    worked out once, as a table, and then looks every pixel up in it. An RGB
    image takes its three tables end to end, 768 entries. The table is clipped
    here as it is built, which makes it the same function as shift_channel
    followed by np.clip, so the two can be compared number for number, and
    with no rounding anywhere they should agree exactly.
    """
    lut = []
    for d in offsets:
        lut += [min(max(v + d, 0), 255) for v in range(256)]
    return np.asarray(img.point(lut), dtype=np.uint8), "img.point(lut)"


# ---------------------------------------------------------------------------
# Web route: POST /channels
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    # whole numbers, held to the sliders' range: at +-255 every value in the
    # channel is already pinned to one end, so there is nowhere further to go
    offsets = [clamp(int(float(request.form.get(k, 0))), -255, 255)
               for k in ("red", "green", "blue")]

    # the same cap as brightness, and for the same reason: adjust.js previews
    # this at exactly this size
    img, original = open_image(file, MAX_DIM)
    source = original.astype(np.float64)

    shifted, elapsed = timed(shift_image, source, offsets)

    # pushed off either end, where the clip pins them
    clipped = 100.0 * float(((shifted < 0.0) | (shifted > 255.0)).mean())
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)

    (library, lib_call), elapsed_lib = timed(library_channels, img, offsets)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        shifted=to_data_uri(shifted),
        library=to_data_uri(library),
        lib_call=lib_call,
        lib_name="Pillow",
        stats=compare(shifted, library),   # same operation, so the numbers mean something
        red=offsets[0],
        green=offsets[1],
        blue=offsets[2],
        rgb_signed=[signed(d) for d in offsets],
        clipped="{:.1f}".format(clipped) if clipped > 0 else None,
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
