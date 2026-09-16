"""Edge detection: keep only the high frequencies of the greyscale image.

How it works
    An edge is where brightness changes quickly, and quick change is high
    frequency. So the image is turned grey, its low frequencies are removed
    with the same continuous Fourier transform sharpen uses
    (common/continuous_ft.py), and the size of what is left is the edge
    strength. It is drawn as dark lines on white.

Compared against
    Pillow's FIND_EDGES, a fixed 3x3 spatial kernel. A different method, so
    there are no number-for-number stats.
"""

import numpy as np
from flask import render_template
from PIL import Image, ImageFilter

from backend.common.continuous_ft import high_pass_detail
from backend.common.limits import MAX_DIM
from backend.common.timing import timed, timing_fields
from backend.common.uploads import form_number, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def to_grayscale(rgb):
    """The image in grey, scaled so its brightest pixel is 1."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    peak = gray.max()
    if peak > 0:
        gray = gray / peak
    return gray


def edge_map(channel, cutoff):
    """Edge strength from 0 to 1, inverted so edges are dark on a white background."""
    edges = np.abs(high_pass_detail(channel, cutoff))
    peak = edges.max()
    if peak > 0:
        edges = edges / peak
    return 1 - edges


def detect_edges(original, cutoff):
    return edge_map(to_grayscale(original), cutoff)


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_edges(img):
    """Pillow's edge detector, which is also not the same method.

    FIND_EDGES is a fixed 3x3 spatial kernel; ours discards low frequencies and
    transforms back. It takes no parameters, so the cutoff slider has nothing to
    map onto. Inverted to match ours, which draws dark lines on white.
    """
    return 255 - np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Web route: POST /edges
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    cutoff = form_number("cutoff", 15)

    img, original = open_image(file, MAX_DIM)

    detected, elapsed = timed(detect_edges, original, cutoff)
    detected = np.clip(detected * 255.0, 0, 255).astype(np.uint8)

    library, elapsed_lib = timed(library_edges, img)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        edges=to_data_uri(detected),
        library=to_data_uri(library),
        lib_call="ImageFilter.FIND_EDGES",
        different_method=True,
        edge_cutoff="{:g}".format(cutoff),
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
