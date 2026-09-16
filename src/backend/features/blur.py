"""Blur: 2D convolution with a small kernel, written as four nested loops.

How it works
    Every output pixel is a weighted average of the k x k pixels around it.
    The weights are the kernel, and its shape decides what kind of blur it is:
    a flat square (box), a bell (Gaussian), a flat disc (lens blur) or a line
    (motion blur). Each colour channel is blurred on its own.

Compared against
    Pillow's BoxBlur for the box, and SciPy for the rest. Both are the same
    operation with the same border handling, so the two results are compared
    number for number.

Mirrored in the browser
    frontend/static/js/conv-viz.js animates this exact convolution on the
    user's own image, and has to reproduce the /blur result byte for byte. Any
    change to the kernels or to convolve2d has to be ported there.
"""

import numpy as np
from flask import render_template, request
from PIL import ImageFilter
from scipy import ndimage   # the comparison baseline only, in library_blur

from backend.common.limits import MAX_DIM
from backend.common.metrics import compare
from backend.common.timing import timed, timing_fields
from backend.common.uploads import open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
#
# Every kernel below is copied into conv-viz.js, which has to reproduce the
# /blur result byte for byte, so each is built from operations JavaScript
# does identically: no np.round (it rounds halves to even, Math.round rounds
# them up) and no ndarray.sum on weights that are not whole numbers (numpy
# sums pairwise, in a different order from a plain loop).

def box_kernel(size):
    """returns a kernel of size x size where each cell = 1 / (size ** 2)"""
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian_kernel(size):
    """A sampled Gaussian, normalised to sum to 1.

    sigma = size / 6, so the kernel spans +-3 sigma. Past that the weights are
    under 1% of the centre, so cutting there costs almost nothing, while a
    wider sigma in the same box would be chopped off square at the edges. The
    one size slider therefore sets the strength, just as it does for the box.

    The sum is a plain loop in row order so the browser can match it exactly.
    The only thing it cannot match is exp itself, which can differ in the last
    bit between numpy and Math.exp -- enough, very rarely, to move a truncated
    pixel by one level.
    """
    sigma = size / 6.0
    r = size // 2
    d = np.arange(size) - r
    g = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2.0 * sigma ** 2))
    total = 0.0
    for v in g.flat:
        total += v
    return g / total


def disk_kernel(size):
    """A flat disc: equal weight on every cell within size // 2 of the centre.

    What an out-of-focus lens does to a point of light, which is why this is
    the "circular" or lens blur. The test is on whole-number squared distances,
    so nothing is rounded anywhere. At 3x3 the disc of radius 1 is a plus sign.
    """
    r = size // 2
    d = np.arange(size) - r
    inside = (d[:, None] ** 2 + d[None, :] ** 2) <= r * r
    return inside / float(np.count_nonzero(inside))


def motion_kernel(size, angle):
    """A line of equal weights through the centre, `angle` degrees anticlockwise
    from horizontal: what a camera moving in a straight line during the
    exposure does to every point.

    `size` points are stepped along the line one cell apart, and each lands on
    its nearest cell. Rounding is floor(v + 0.5) rather than np.round, and the
    1e-9 nudge keeps a value that should be exactly half-way (at 60 degrees
    some are) on the same side in numpy and in the browser, whatever the last
    bit of cos or sin came out as. The angle is converted as angle * pi / 180
    for the same reason: that is the order of operations the JS copy uses.
    """
    r = size // 2
    theta = angle * np.pi / 180.0
    kernel = np.zeros((size, size), dtype=np.float64)
    for t in range(-r, r + 1):
        x = int(np.floor(r + t * np.cos(theta) + 0.5 + 1e-9))
        y = int(np.floor(r - t * np.sin(theta) + 0.5 + 1e-9))   # rows run downwards
        kernel[y, x] = 1.0
    return kernel / np.count_nonzero(kernel)


# the kinds the blur form offers (the `kernel` field), and what the page calls each
KERNELS = {"box": "box", "gaussian": "Gaussian", "disk": "circular", "motion": "motion"}


def make_kernel(kind, size, angle=0.0):
    """The kernel for a form's choice. Anything unrecognised gets the box."""
    if kind == "gaussian":
        return gaussian_kernel(size)
    if kind == "disk":
        return disk_kernel(size)
    if kind == "motion":
        return motion_kernel(size, angle)
    return box_kernel(size)


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def convolve2d(channel, kernel):
    """2D convolution, written out by hand.

    For every output pixel we sum the neighbourhood weighted by the kernel:

        out[y, x] = sum_ky sum_kx  padded[y + ky, x + kx] * kernel[ky, kx]

    The kernel is flipped first, which is what makes this convolution rather
    than correlation. (For a symmetric kernel like the box filter the two are
    identical, but the flip is part of the definition.)
    """
    kernel = kernel[::-1, ::-1]

    k_height, k_width = kernel.shape
    p_height, p_width = k_height // 2, k_width // 2

    # Pad by replicating the border so the output keeps the input's size.
    padded = np.pad(channel, ((p_height, p_height), (p_width, p_width)), mode="edge")

    h, w = channel.shape
    out = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            total = 0.0
            for ky in range(k_height):
                for kx in range(k_width):
                    total += padded[y + ky, x + kx] * kernel[ky, kx]
            out[y, x] = total

    return out


def blur_image(source, kernel):
    """Convolve each colour channel of a float RGB image independently."""
    blurred = np.zeros_like(source)
    for c in range(3):
        blurred[:, :, c] = convolve2d(source[:, :, c], kernel)
    return blurred


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_blur(img, source, kind, ksize, kernel):
    """The same blur from a library, as a check on convolve2d.

    Returns the image, the call as the page prints it, and the library's name.

    The box goes to Pillow. BoxBlur's radius counts pixels either side of the
    centre, so a k x k kernel is radius (k - 1) / 2. It pads by replicating the
    border, exactly as convolve2d does, so the two really are the same operation
    and can be compared number for number.

    The rest go to SciPy, because Pillow cannot do them: its general Kernel
    filter stops at 5x5. The Gaussian uses gaussian_filter, SciPy's own blur,
    with its radius pinned to k // 2 -- it then builds the same normalised
    Gaussian we do, only separably, as the product of two 1D ones, which is the
    same kernel. Circular and motion have no named SciPy function, so they are
    ndimage.convolve with our kernel. mode="nearest" is edge replication, as in
    convolve2d, and the float result is truncated to uint8 the same way, so
    what is left to differ is only the order the sums were done in.
    """
    if kind == "box":
        radius = (ksize - 1) / 2
        return (
            np.asarray(img.filter(ImageFilter.BoxBlur(radius)), dtype=np.uint8),
            "ImageFilter.BoxBlur({:g})".format(radius),
            "Pillow",
        )

    out = np.zeros_like(source)
    if kind == "gaussian":
        sigma = ksize / 6.0
        for c in range(3):
            out[:, :, c] = ndimage.gaussian_filter(source[:, :, c], sigma,
                                                   mode="nearest", radius=ksize // 2)
        call = "ndimage.gaussian_filter(sigma={:.3g}, radius={}, mode=\"nearest\")".format(sigma, ksize // 2)
    else:
        for c in range(3):
            out[:, :, c] = ndimage.convolve(source[:, :, c], kernel, mode="nearest")
        call = "ndimage.convolve(kernel, mode=\"nearest\")"

    return np.clip(out, 0, 255).astype(np.uint8), call, "SciPy"


# ---------------------------------------------------------------------------
# Web route: POST /blur
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    ksize = int(request.form.get("ksize", 5))
    kind = request.form.get("kernel", "box")
    if kind not in KERNELS:
        kind = "box"
    angle = float(request.form.get("angle", 0))

    img, original = open_image(file, MAX_DIM)

    # every kind goes through the same dense k x k loop, zeros included, so the
    # timings in the MAX_DIM comment hold for all of them
    kernel = make_kernel(kind, ksize, angle)
    source = original.astype(np.float64)

    blurred, elapsed = timed(blur_image, source, kernel)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    # the same operation from a library, as a check on the loop above
    (library, lib_call, lib_name), elapsed_lib = timed(library_blur, img, source, kind, ksize, kernel)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        blurred=to_data_uri(blurred),
        library=to_data_uri(library),
        lib_call=lib_call,
        lib_name=lib_name,
        stats=compare(blurred, library),   # same operation, so the numbers mean something
        ksize=ksize,
        kernel_kind=kind,
        kernel_name=KERNELS[kind],
        angle="{:g}".format(angle),
        sigma="{:.3g}".format(ksize / 6.0),
        # the weights themselves, for the Method panel
        kernel_rows=[["{:.4f}".format(v) for v in row] for row in kernel],
        size=size_text(original),
        **timing_fields(elapsed, elapsed_lib),
    )
