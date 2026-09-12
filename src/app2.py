import base64
import io
import time

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage   # comparison baselines only, in library_blur

import compress
import denoise

app = Flask(__name__)

# Images are capped to this size because the convolution below is a literal
# per-pixel Python loop, so cost grows with the pixel count times the kernel
# area. At 500px a 3x3 blur takes ~0.9s and an 11x11 blur takes ~10s, which is
# why the kernel slider in the template stops at 11.
MAX_DIM = 500

# The template sizes the pre-run preview to match what the processed image will
# be, so it needs the same number. Exposing it here keeps the two in step.
app.jinja_env.globals["MAX_DIM"] = MAX_DIM


def box_kernel(size):
    """returns a kernel of size x size where each cell = 1 / (size ** 2)"""
    return np.ones((size, size), dtype=np.float64) / (size * size)


# Every kernel below is copied into conv-viz.js, which has to reproduce the
# /blur result byte for byte, so each is built from operations JavaScript
# does identically: no np.round (it rounds halves to even, Math.round rounds
# them up) and no ndarray.sum on weights that are not whole numbers (numpy
# sums pairwise, in a different order from a plain loop).

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


# the kinds the blur form offers, and what the page calls each
KERNELS = {"box": "box", "gaussian": "Gaussian", "disk": "circular", "motion": "motion"}


def make_kernel(kind, size, angle=0.0):
    if kind == "gaussian":
        return gaussian_kernel(size)
    if kind == "disk":
        return disk_kernel(size)
    if kind == "motion":
        return motion_kernel(size, angle)
    return box_kernel(size)


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


def spatial_axes(height, width):
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    return x, y


def frequency_axes(x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), x.size)
    v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), y.size)
    return u, v


def compute_cft(channel, x, y, u, v):
    r1 = np.zeros((y.size, u.size), dtype=np.float64)
    r2 = np.zeros((y.size, u.size), dtype=np.float64)
    for j in range(u.size):
        theta = 2 * np.pi * u[j] * x
        r1[:, j] = np.trapezoid(channel * np.cos(theta), x, axis=1)
        r2[:, j] = np.trapezoid(channel * np.sin(theta), x, axis=1)

    real = np.zeros((v.size, u.size), dtype=np.float64)
    imag = np.zeros((v.size, u.size), dtype=np.float64)
    for i in range(v.size):
        theta = 2 * np.pi * v[i] * y
        cos_v_y = np.cos(theta)[:, None]
        sin_v_y = np.sin(theta)[:, None]
        real[i, :] = np.trapezoid(r1 * cos_v_y - r2 * sin_v_y, y, axis=0)
        imag[i, :] = -np.trapezoid(r2 * cos_v_y + r1 * sin_v_y, y, axis=0)

    return real, imag


def high_pass(real, imag, cutoff, u, v):
    """Drop the low frequencies, keeping what changes quickly.

    Two things here are easy to get wrong, and both were:

    1. The radius is measured off the u and v axes rather than off array
       indices. frequency_axes returns linspace(-Nyquist, +Nyquist, N), which
       for an even N has no exact zero bin -- u[N//2] sits half a bin above
       zero. Centring the mask on index N//2 therefore centres it half a bin
       off DC, which tilts the whole filter.

    2. The edge is a Gaussian, not a step. A hard circle in the frequency plane
       is a sinc in the image plane, so every edge comes back wrapped in
       ripples that spread across the entire picture -- the Gibbs phenomenon.
       That is what made the old output look nothing like an unsharp mask.
       Fading out over `cutoff` instead removes the ripples entirely.

    The mask is 1 - exp(-D^2 / 2c^2): a Gaussian low-pass subtracted from
    everything, which is precisely unsharp masking done in the frequency
    domain. `cutoff` is in bins, so it keeps the meaning it had before.
    """
    du = u[1] - u[0]
    dv = v[1] - v[0]
    d = np.sqrt((v[:, None] / dv) ** 2 + (u[None, :] / du) ** 2)
    mask = 1.0 - np.exp(-(d ** 2) / (2.0 * max(cutoff, 1e-6) ** 2))
    return real * mask, imag * mask


def reconstruct(real, imag, u, v, x, y):
    r1 = np.zeros((y.size, u.size), dtype=np.float64)
    r2 = np.zeros((y.size, u.size), dtype=np.float64)
    for i in range(y.size):
        theta = 2 * np.pi * v * y[i]
        cos_v_y = np.cos(theta)[:, None]
        sin_v_y = np.sin(theta)[:, None]
        r1[i, :] = np.trapezoid(real * cos_v_y - imag * sin_v_y, v, axis=0)
        r2[i, :] = np.trapezoid(real * sin_v_y + imag * cos_v_y, v, axis=0)

    image = np.zeros((y.size, x.size), dtype=np.float64)
    for j in range(x.size):
        theta = 2 * np.pi * x[j] * u
        image[:, j] = np.trapezoid(r1 * np.cos(theta) - r2 * np.sin(theta), u, axis=1)

    return image


def high_pass_detail(channel, cutoff):
    x, y = spatial_axes(channel.shape[0], channel.shape[1])
    u, v = frequency_axes(x, y)
    real, imag = compute_cft(channel, x, y, u, v)
    real, imag = high_pass(real, imag, cutoff, u, v)
    return reconstruct(real, imag, u, v, x, y)


def to_grayscale(rgb):
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    peak = gray.max()
    if peak > 0:
        gray = gray / peak
    return gray


def edge_map(channel, cutoff):
    edges = np.abs(high_pass_detail(channel, cutoff))
    peak = edges.max()
    if peak > 0:
        edges = edges / peak
    return 1 - edges


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


def library_edges(img):
    """Pillow's edge detector, which is also not the same method.

    FIND_EDGES is a fixed 3x3 spatial kernel; ours discards low frequencies and
    transforms back. It takes no parameters, so the cutoff slider has nothing to
    map onto. Inverted to match ours, which draws dark lines on white.
    """
    return 255 - np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8)


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


def library_jpeg(img, target_bytes):
    """Pillow's JPEG, with the quality searched until the file is about as big as ours.

    A different method (8x8 DCT blocks, quantisation, Huffman coding), so the
    only fair comparison is at equal size: then the PSNR says which kept more
    of the picture for the same bytes. File size rises with quality closely
    enough that a binary search finds the nearest within about 7 saves. Note
    the JPEG figure includes its headers (~600 bytes) and ours is payload only,
    which matters at the smallest settings, where even quality 1 can be bigger.

    The search runs to 100 rather than Pillow's recommended 95: past 95 JPEG
    gains little, but our high settings are bigger than a quality-95 file, and
    stopping there would leave those comparisons at mismatched sizes.
    """
    def save(quality):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    lo, hi = 1, 100
    best = None
    while lo <= hi:
        quality = (lo + hi) // 2
        data = save(quality)
        if best is None or abs(len(data) - target_bytes) < abs(len(best[1]) - target_bytes):
            best = (quality, data)
        if len(data) < target_bytes:
            lo = quality + 1
        else:
            hi = quality - 1

    quality, data = best
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    return decoded, len(data), quality


def compare(a, b):
    """How far apart two uint8 images are. Only meaningful when the two were
    produced by the same operation, which is true of the blur and the two
    point operations (brightness, channels) and nothing else."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "max": int(d.max()),
        "within1": "{:.1f}".format(100.0 * float((d <= 1).mean())),
    }


def to_data_uri(arr):
    """Encode a uint8 image array as a base64 PNG so it can go straight into <img>."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def spectrum_plate(mag, mask, view):
    """Draw the spectrum, with the mask shown over it.

    Log scaled, because the centre bin is orders of magnitude above everything
    else and a linear scale is a single white dot on black.
    """
    s = denoise.spectrum_picture(mag)

    if view == "spectrum":
        rgb = np.stack([s] * 3, axis=-1)
    elif view == "removed":
        rgb = np.stack([s * (1.0 - mask)] * 3, axis=-1)
    else:
        # kept frequencies stay grey; discarded ones are tinted, floored so they
        # are visible even out where the spectrum is nearly black
        accent = np.array([0.04, 0.52, 1.0])
        keep = mask[..., None]
        gone = (1.0 - mask)[..., None]
        rgb = np.stack([s] * 3, -1) * keep + accent * gone * np.maximum(s, 0.28)[..., None]

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def numpy_filter(rgb, mask):
    """The same mask applied with numpy's FFT instead of ours.

    NOTE: this is the only numpy.fft call in the project, it lives here in the
    web app rather than in transforms.py, and it exists purely as a check on the
    hand-written transform. Delete this function and the `library=` argument
    below if the assignment forbids the import anywhere in the tree.
    """
    padded, (h, w) = denoise.pad_to_pow2(rgb)
    out = np.zeros_like(padded)
    for c in range(3):
        spec = np.fft.fftshift(np.fft.fft2(padded[:, :, c]))
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask)))
    return np.clip(out[:h, :w], 0.0, 1.0)


def psnr(a, b):
    """Peak signal to noise, in dB, for two float images in 0..1."""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return None
    return "{:.1f}".format(10.0 * np.log10(1.0 / mse))


def kilobytes(n):
    return "{:.1f} KB".format(n / 1024.0)


def signed(n):
    """An offset as the page shows it: +40, −20 (a true minus sign) or 0."""
    if n > 0:
        return "+{}".format(n)
    if n < 0:
        return "−{}".format(-n)
    return "0"


def pixel_loss(a, b):
    """Average error of a channel value, as a percentage of the full 0..255 range.

    The "% loss" in the compression stats, with "% quality" as 100 minus it.
    It reads gently -- a few percent can already be visible ringing -- which is
    why PSNR is shown beside it.
    """
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return 100.0 * float(np.mean(diff)) / 255.0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blur", methods=["POST"])
def blur():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    ksize = int(request.form.get("ksize", 5))
    kind = request.form.get("kernel", "box")
    if kind not in KERNELS:
        kind = "box"
    angle = float(request.form.get("angle", 0))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    img.thumbnail((MAX_DIM, MAX_DIM))
    original = np.asarray(img, dtype=np.uint8)

    # every kind goes through the same dense k x k loop, zeros included, so the
    # timings in the MAX_DIM comment hold for all of them
    kernel = make_kernel(kind, ksize, angle)
    source = original.astype(np.float64)

    start = time.time()
    blurred = np.zeros_like(source)
    for c in range(3):  # convolve each colour channel independently
        blurred[:, :, c] = convolve2d(source[:, :, c], kernel)
    elapsed = time.time() - start

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    # the same operation from a library, as a check on the loop above
    start = time.time()
    library, lib_call, lib_name = library_blur(img, source, kind, ksize, kernel)
    elapsed_lib = time.time() - start

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
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


@app.route("/sharpen", methods=["POST"])
def sharpen():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    cutoff = float(request.form.get("cutoff", 15))
    amount = float(request.form.get("amount", 1.0))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    img.thumbnail((MAX_DIM, MAX_DIM))
    original = np.asarray(img, dtype=np.uint8)

    source = original.astype(np.float64) / 255.0

    start = time.time()
    sharpened = np.zeros_like(source)
    for c in range(3):
        detail = high_pass_detail(source[:, :, c], cutoff)
        sharpened[:, :, c] = source[:, :, c] + amount * detail
    elapsed = time.time() - start

    sharpened = np.clip(sharpened * 255.0, 0, 255).astype(np.uint8)

    start = time.time()
    library, lib_call = library_sharpen(img, cutoff, amount)
    elapsed_lib = time.time() - start

    return render_template(
        "index.html",
        original=to_data_uri(original),
        sharpened=to_data_uri(sharpened),
        library=to_data_uri(library),
        lib_call=lib_call,
        different_method=True,            # no numbers: it is not the same algorithm
        cutoff="{:g}".format(cutoff),
        amount="{:g}".format(amount),
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


@app.route("/edges", methods=["POST"])
def edges():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    cutoff = float(request.form.get("cutoff", 15))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    img.thumbnail((MAX_DIM, MAX_DIM))
    original = np.asarray(img, dtype=np.uint8)

    start = time.time()
    detected = edge_map(to_grayscale(original), cutoff)
    elapsed = time.time() - start

    detected = np.clip(detected * 255.0, 0, 255).astype(np.uint8)

    start = time.time()
    library = library_edges(img)
    elapsed_lib = time.time() - start

    return render_template(
        "index.html",
        original=to_data_uri(original),
        edges=to_data_uri(detected),
        library=to_data_uri(library),
        lib_call="ImageFilter.FIND_EDGES",
        different_method=True,
        edge_cutoff="{:g}".format(cutoff),
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


@app.route("/brightness", methods=["POST"])
def brightness():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    # the same range as the slider: at 2x everything brighter than mid-grey is
    # already clipped white, so further up there is little image left
    gain = min(max(float(request.form.get("gain", 1.0)), 0.0), 2.0)

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    # one multiply per value, so this would cope with far more than MAX_DIM,
    # but the live preview in the browser reproduces this exact cap
    img.thumbnail((MAX_DIM, MAX_DIM))
    original = np.asarray(img, dtype=np.uint8)
    source = original.astype(np.float64)

    start = time.time()
    brightened = np.zeros_like(source)
    for c in range(3):  # the same gain on every channel, so the hue is kept
        brightened[:, :, c] = brighten(source[:, :, c], gain)
    elapsed = time.time() - start

    # the share pushed past 255, which the clip flattens for good
    clipped = 100.0 * float((brightened > 255.0).mean())
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)

    start = time.time()
    library, lib_call = library_brightness(img, gain)
    elapsed_lib = time.time() - start

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
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


@app.route("/channels", methods=["POST"])
def channels():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    # whole numbers, held to the sliders' range: at +-255 every value in the
    # channel is already pinned to one end, so there is nowhere further to go
    offsets = [min(max(int(float(request.form.get(k, 0))), -255), 255)
               for k in ("red", "green", "blue")]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    # the same cap as brightness, and for the same reason: adjust.js previews
    # this at exactly this size
    img.thumbnail((MAX_DIM, MAX_DIM))
    original = np.asarray(img, dtype=np.uint8)
    source = original.astype(np.float64)

    start = time.time()
    shifted = np.zeros_like(source)
    for c in range(3):  # each channel its own offset, which is what moves the hue
        shifted[:, :, c] = shift_channel(source[:, :, c], offsets[c])
    elapsed = time.time() - start

    # pushed off either end, where the clip pins them
    clipped = 100.0 * float(((shifted < 0.0) | (shifted > 255.0)).mean())
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)

    start = time.time()
    library, lib_call = library_channels(img, offsets)
    elapsed_lib = time.time() - start

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
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


@app.route("/denoise", methods=["POST"])
def denoise_view():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    kind = request.form.get("noise", "periodic")
    amount = float(request.form.get("noise_amount", 0.3))
    which = request.form.get("filter", "notch")
    cutoff = float(request.form.get("fcut", 45))
    width = float(request.form.get("nwidth", 3))
    softness = float(request.form.get("softness", 20))
    view = request.form.get("specview", "mask")

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    img.thumbnail((denoise.WORK_DIM, denoise.WORK_DIM))
    original = np.asarray(img, dtype=np.uint8)
    clean = original.astype(np.float64) / 255.0

    noisy = denoise.add_noise(clean, kind, amount, seed=0)

    # the spectrum is taken of the luminance: one picture to look at, and one
    # place to hunt for peaks, while the mask itself is applied to all three
    # colour channels
    padded, _ = denoise.pad_to_pow2(noisy)
    start = time.time()
    mag = np.abs(denoise.shift(denoise.fft2(denoise.luma(padded))))

    peaks = []
    if which == "notch":
        mask, peaks = denoise.mask_notch(mag, max(8.0, width * 2), width, softness)
    elif which == "gaussian":
        mask = denoise.mask_gaussian(mag.shape[0], mag.shape[1], cutoff, softness)
    else:
        mask = denoise.mask_ideal(mag.shape[0], mag.shape[1], cutoff)

    cleaned = denoise.filter_channels(noisy, mask)
    elapsed = time.time() - start

    start = time.time()
    library = numpy_filter(noisy, mask)
    elapsed_lib = time.time() - start

    kept = "{:.1f}".format(100.0 * float(np.mean(mask)))

    return render_template(
        "index.html",
        original=to_data_uri(original),
        noisy=to_data_uri((noisy * 255).astype(np.uint8)),
        spectrum=to_data_uri(spectrum_plate(mag, mask, view)),
        denoised=to_data_uri((cleaned * 255).astype(np.uint8)),
        library=to_data_uri((library * 255).astype(np.uint8)),
        lib_call="numpy.fft.fft2 / ifft2",
        stats=compare((cleaned * 255).astype(np.uint8), (library * 255).astype(np.uint8)),
        noise=kind,
        noise_amount="{:g}".format(amount),
        filt=which,
        fcut="{:g}".format(cutoff),
        nwidth="{:g}".format(width),
        softness="{:g}".format(softness),
        specview=view,
        peaks=len(peaks),
        kept=kept,
        psnr_noisy=psnr(clean, noisy),
        psnr_clean=psnr(clean, cleaned),
        size="{} x {}".format(original.shape[1], original.shape[0]),
        spec_size="{} x {}".format(mag.shape[1], mag.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


@app.route("/compress", methods=["POST"])
def compress_view():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    # capped at 25%, the same as the slider. A kept bin costs about 7 bytes
    # against 3 per raw pixel, so 25% is only ~1.7x smaller and break-even is
    # near 43%; past the cap the "compressed" file stops earning the name.
    keep = min(max(float(request.form.get("keep", 5)), 0.5), 25.0)

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    # the same cap as denoise, and for the same reason: the transform is a
    # Python butterfly loop. It also keeps the padded spectrum within 65536
    # bins, which is what lets compress.py store positions as uint16.
    img.thumbnail((denoise.WORK_DIM, denoise.WORK_DIM))
    original = np.asarray(img, dtype=np.uint8)
    clean = original.astype(np.float64) / 255.0

    start = time.time()
    packed, mag = compress.encode(clean, keep / 100.0)
    elapsed_enc = time.time() - start

    start = time.time()
    restored = compress.decode(packed)
    elapsed_dec = time.time() - start

    # rounded to uint8 before measuring, so both PSNR figures are taken on the
    # 8-bit images actually shown, and neither side gets a precision advantage
    restored = np.round(restored * 255.0).astype(np.uint8)

    bytes_raw = original.size
    bytes_ours = compress.packed_bytes(packed)
    mask = compress.kept_mask(packed)

    start = time.time()
    library, bytes_lib, jpeg_q = library_jpeg(img, bytes_ours)
    elapsed_lib = time.time() - start

    elapsed = elapsed_enc + elapsed_dec

    # When the sizes could not be matched, the note has to say so. Past about
    # 7% kept our file is bigger than anything JPEG produces (quality 100), and
    # at the very bottom JPEG's smallest file, headers included, can be bigger
    # than ours.
    if jpeg_q == 100 and bytes_lib < 0.85 * bytes_ours:
        jpeg_fit = "ceiling"
    elif jpeg_q == 1 and bytes_lib > 1.15 * bytes_ours:
        jpeg_fit = "floor"
    else:
        jpeg_fit = None

    loss_ours = pixel_loss(original, restored)
    loss_lib = pixel_loss(original, library)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        compressed=to_data_uri(restored),
        spectrum=to_data_uri(spectrum_plate(denoise.shift(mag), denoise.shift(mask), "mask")),
        library=to_data_uri(library),
        lib_call="Image.save(format=\"JPEG\", quality={})".format(jpeg_q),
        different_method=True,            # not the same algorithm: compared by PSNR at equal size
        keep="{:g}".format(keep),
        kept="{:.1f}".format(100.0 * float(np.mean(mask))),
        stored=int(packed["idx"].size),
        bytes_raw=kilobytes(bytes_raw),
        bytes_ours=kilobytes(bytes_ours),
        bytes_lib=kilobytes(bytes_lib),
        ratio="{:.1f}".format(bytes_raw / bytes_ours),
        ratio_lib="{:.1f}".format(bytes_raw / bytes_lib),
        psnr_ours=psnr(clean, restored / 255.0),
        psnr_lib=psnr(clean, library / 255.0),
        saved_ours="{:.1f}".format(100.0 * (1.0 - bytes_ours / bytes_raw)),
        saved_lib="{:.1f}".format(100.0 * (1.0 - bytes_lib / bytes_raw)),
        loss_ours="{:.2f}".format(loss_ours),
        loss_lib="{:.2f}".format(loss_lib),
        quality_ours="{:.2f}".format(100.0 - loss_ours),
        quality_lib="{:.2f}".format(100.0 - loss_lib),
        jpeg_q=jpeg_q,
        jpeg_fit=jpeg_fit,
        size="{} x {}".format(original.shape[1], original.shape[0]),
        spec_size="{} x {}".format(mag.shape[1], mag.shape[0]),
        elapsed="{:.2f}".format(elapsed),
        elapsed_enc="{:.2f}".format(elapsed_enc),
        elapsed_dec="{:.2f}".format(elapsed_dec),
        elapsed_lib="{:.4f}".format(elapsed_lib),
        speedup="{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    )


if __name__ == "__main__":
    app.run(debug=True)
