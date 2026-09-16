"""Filters: grayscale, invert, black & white, pixelate and vintage, each written as a loop.

How it works
    Every filter visits every pixel with two plain loops, one for the row and
    one for the column, and works out the new colour from the old one:

        grayscale        mix each channel towards the pixel's grey
        invert           mix each channel towards 255 minus itself
        black & white    white if the pixel's grey reaches a threshold, else black
        pixelate         cut the image into square tiles, paint each with its average
        vintage          mix each channel towards a sepia tone

    Four of them only look at the pixel itself (point operations, like
    brightness). Pixelate is the exception: a tile's colour depends on all the
    pixels in the tile.

    Each filter returns floats; the route clips them to 0..255 and truncates to
    whole numbers, exactly as brightness does.

Compared against
    Pillow doing the same job (blend with a greyscale or inverted copy, a
    threshold lookup, Image.reduce, a colour matrix). They are the same
    operations, so the two are compared number for number; Pillow rounds where
    we truncate, so a difference of 1 here and there is expected.

Mirrored in the browser
    frontend/static/js/adjust.js repeats these loops for the live preview, so a
    change here has to be ported there.
"""

import numpy as np
from flask import render_template, request
from PIL import Image, ImageOps

from backend.common.limits import MAX_DIM
from backend.common.metrics import compare
from backend.common.timing import timed, timing_fields
from backend.common.uploads import clamp, form_number, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm: one function per filter
# ---------------------------------------------------------------------------
#
# All of them take the image as a float array of shape (height, width, 3),
# with values 0..255, and return a new array of the same shape.

def gray_of(r, g, b):
    """How bright a colour looks, as one number: ITU-R 601-2 luma.

    Green counts most and blue least, because that is how sensitive the eye
    is to each. These are the same weights Pillow's convert("L") uses.
    """
    return 0.299 * r + 0.587 * g + 0.114 * b


def grayscale(image, amount):
    """Mix every channel towards the pixel's grey. amount 0 = unchanged, 1 = fully grey."""
    height, width, _ = image.shape
    out = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            gray = gray_of(r, g, b)
            for c in range(3):
                out[y, x, c] = image[y, x, c] + amount * (gray - image[y, x, c])

    return out


def invert(image, amount):
    """Mix every channel towards its opposite, 255 - value. amount 1 = a full negative."""
    height, width, _ = image.shape
    out = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            for c in range(3):
                opposite = 255 - image[y, x, c]
                out[y, x, c] = image[y, x, c] + amount * (opposite - image[y, x, c])

    return out


def black_and_white(image, threshold):
    """Every pixel becomes pure white or pure black, depending on its grey."""
    height, width, _ = image.shape
    out = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            if gray_of(r, g, b) >= threshold:
                out[y, x] = 255
            else:
                out[y, x] = 0

    return out


def pixelate(image, block):
    """Cut the image into block x block tiles and paint each tile with its average colour.

    Tiles on the right and bottom edges are smaller when the size does not
    divide evenly; they are averaged over the pixels they actually contain.
    """
    height, width, _ = image.shape
    out = np.zeros((height, width, 3))

    # the top-left corner of each tile
    for top in range(0, height, block):
        for left in range(0, width, block):
            bottom = min(top + block, height)
            right = min(left + block, width)

            # add up every pixel in the tile, one channel at a time
            total = [0.0, 0.0, 0.0]
            for y in range(top, bottom):
                for x in range(left, right):
                    for c in range(3):
                        total[c] += image[y, x, c]
            count = (bottom - top) * (right - left)

            # then paint the whole tile with the average
            for y in range(top, bottom):
                for x in range(left, right):
                    for c in range(3):
                        out[y, x, c] = total[c] / count

    return out


def sepia_of(r, g, b):
    """The classic sepia tone of a colour: a warm brown version of it, each channel capped at 255."""
    sepia_r = min(255.0, 0.393 * r + 0.769 * g + 0.189 * b)
    sepia_g = min(255.0, 0.349 * r + 0.686 * g + 0.168 * b)
    sepia_b = min(255.0, 0.272 * r + 0.534 * g + 0.131 * b)
    return sepia_r, sepia_g, sepia_b


def vintage(image, amount):
    """Mix every pixel towards its sepia tone. amount 1 = full sepia."""
    height, width, _ = image.shape
    out = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            sepia = sepia_of(r, g, b)
            for c in range(3):
                out[y, x, c] = image[y, x, c] + amount * (sepia[c] - image[y, x, c])

    return out


# The filters the form offers. For each: what the page calls it, the function,
# and its one slider -- the form field's name, its default and its range.
FILTERS = {
    "grayscale":       {"name": "Grayscale",     "function": grayscale,       "field": "gray_amount",    "default": 1.0, "range": (0.0, 1.0)},
    "invert":          {"name": "Invert",        "function": invert,          "field": "invert_amount",  "default": 1.0, "range": (0.0, 1.0)},
    "black_and_white": {"name": "Black & white", "function": black_and_white, "field": "threshold",      "default": 128, "range": (0, 255)},
    "pixelate":        {"name": "Pixelate",      "function": pixelate,        "field": "block",          "default": 8,   "range": (2, 32)},
    "vintage":         {"name": "Vintage",       "function": vintage,         "field": "vintage_amount", "default": 1.0, "range": (0.0, 1.0)},
}


def setting_text(kind, setting):
    """The slider's value as the page shows it: "amount 80%", "threshold 128", "8 px blocks"."""
    if kind == "black_and_white":
        return "threshold {}".format(setting)
    if kind == "pixelate":
        return "{} px blocks".format(setting)
    return "amount {:.0f}%".format(setting * 100)


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

# Sepia as a colour matrix for Image.convert: each row is (r, g, b, offset)
SEPIA_MATRIX = (
    0.393, 0.769, 0.189, 0,
    0.349, 0.686, 0.168, 0,
    0.272, 0.534, 0.131, 0,
)


def library_filter(img, kind, setting):
    """The same filter from Pillow. Returns the image and the call as the page prints it.

    The "amount" filters are Image.blend between the picture and the fully
    filtered picture, which is the same mix as ours: a + t * (b - a).
    """
    if kind == "grayscale":
        gray = img.convert("L").convert("RGB")
        result = Image.blend(img, gray, setting)
        call = "Image.blend(img, img.convert(\"L\"), {:g})".format(setting)
    elif kind == "invert":
        result = Image.blend(img, ImageOps.invert(img), setting)
        call = "Image.blend(img, ImageOps.invert(img), {:g})".format(setting)
    elif kind == "black_and_white":
        # a lookup table: every grey value from 0 to 255 mapped to black or white
        table = [255 if v >= setting else 0 for v in range(256)]
        result = img.convert("L").point(table).convert("RGB")
        call = "img.convert(\"L\").point(threshold {})".format(setting)
    elif kind == "pixelate":
        # reduce averages each block x block tile into one pixel; NEAREST then
        # blows every pixel back up into a solid tile, and crop trims the
        # overhang where the edge tiles were smaller
        small = img.reduce(setting)
        big = small.resize((small.width * setting, small.height * setting), Image.Resampling.NEAREST)
        result = big.crop((0, 0, img.width, img.height))
        call = "img.reduce({0}).resize(×{0}, NEAREST)".format(setting)
    else:
        sepia = img.convert("RGB", SEPIA_MATRIX)
        result = Image.blend(img, sepia, setting)
        call = "Image.blend(img, img.convert(\"RGB\", sepia), {:g})".format(setting)

    return np.asarray(result, dtype=np.uint8), call


# ---------------------------------------------------------------------------
# Web route: POST /filters
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    kind = request.form.get("filter_kind", "grayscale")
    if kind not in FILTERS:
        kind = "grayscale"
    chosen = FILTERS[kind]

    # the chosen filter's slider, held to the slider's range; threshold and
    # block size are whole numbers
    low, high = chosen["range"]
    setting = clamp(form_number(chosen["field"], chosen["default"]), low, high)
    if kind in ("black_and_white", "pixelate"):
        setting = int(setting)

    img, original = open_image(file, MAX_DIM)
    source = original.astype(np.float64)

    filtered, elapsed = timed(chosen["function"], source, setting)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    (library, lib_call), elapsed_lib = timed(library_filter, img, kind, setting)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        filtered=to_data_uri(filtered),
        library=to_data_uri(library),
        lib_call=lib_call,
        lib_name="Pillow",
        stats=compare(filtered, library),   # same operation, so the numbers mean something
        filter_kind=kind,
        filter_name=chosen["name"],
        filter_setting=setting_text(kind, setting),
        # the slider's value, to put the slider back where it was
        **{chosen["field"]: "{:g}".format(setting)},
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
