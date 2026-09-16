"""Resize: make the image bigger or smaller with nearest-neighbour sampling.

How it works
    The new image gets its own grid of pixels. For each new pixel we ask
    "which old pixel sits under the middle of me?" and copy that pixel's colour.
    Nothing is blended or averaged, which is why an enlarged image looks blocky
    and a shrunk one can look a little jagged.

    Walking along a row, the middle of new pixel 0 is half a step into the old
    image, and every next new pixel is one more step further on, where

        step = old size / new size

    (a step of 2 means every other old pixel is used, a step of 0.5 means every
    old pixel is used twice). The whole number part of that position is the old
    pixel to copy. The same is done down the columns.

Compared against
    Pillow's img.resize(..., NEAREST), which walks the pixels in exactly this
    way, so the two are compared number for number and should agree exactly.

Mirrored in the browser
    frontend/static/js/adjust.js repeats this for the live preview, so a change
    here has to be ported there.
"""

import numpy as np
from flask import render_template
from PIL import Image

from backend.common.limits import MAX_DIM
from backend.common.metrics import compare
from backend.common.timing import timed, timing_fields
from backend.common.uploads import clamp, form_number, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def new_size(old_size, scale):
    """The size after scaling, rounded to a whole number of pixels (at least 1)."""
    return max(1, int(old_size * scale + 0.5))


def old_positions(old_size, new_size):
    """For each new pixel along one side, the old pixel under its middle.

    Starts half a step in and adds one step per pixel. The positions are added
    up rather than worked out as i * step, because that is how Pillow does it,
    and adding up small rounding errors makes a different pixel win now and
    then; doing the same keeps the two identical.
    """
    step = old_size / new_size
    position = step * 0.5
    positions = []
    for i in range(new_size):
        positions.append(int(position))
        position += step
    return positions


def resize_nearest(image, scale):
    """Resize an (height, width, 3) image by `scale` (0.5 = half size, 2 = double)."""
    old_height, old_width, _ = image.shape
    height = new_size(old_height, scale)
    width = new_size(old_width, scale)

    # which old row and old column each new row and new column copies from
    rows = old_positions(old_height, height)
    columns = old_positions(old_width, width)

    out = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            out[y, x] = image[rows[y], columns[x]]

    return out


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_resize(img, scale):
    """Pillow's nearest-neighbour resize to the same size."""
    width = new_size(img.width, scale)
    height = new_size(img.height, scale)
    result = img.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(result, dtype=np.uint8), "img.resize(({}, {}), NEAREST)".format(width, height)


# ---------------------------------------------------------------------------
# Web route: POST /resize
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    # the slider is a percentage, 10% to 200%
    percent = clamp(form_number("scale", 50), 10, 200)
    scale = percent / 100.0

    img, original = open_image(file, MAX_DIM)
    source = original.astype(np.float64)

    resized, elapsed = timed(resize_nearest, source, scale)
    resized = resized.astype(np.uint8)   # copied pixels are already whole numbers

    (library, lib_call), elapsed_lib = timed(library_resize, img, scale)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        resized=to_data_uri(resized),
        library=to_data_uri(library),
        lib_call=lib_call,
        lib_name="Pillow",
        stats=compare(resized, library),   # same operation, so the numbers mean something
        scale="{:g}".format(percent),
        new_size=size_text(resized),
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
