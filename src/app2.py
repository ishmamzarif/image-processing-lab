import base64
import io
import time

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Images are capped to this size because the convolution below is a literal
# per-pixel Python loop. 200px keeps a blur under a couple of seconds.
MAX_DIM = 200


def box_kernel(size):
    """returns a kernel of size x size where each cell = 1 / (size ** 2)"""
    return np.ones((size, size), dtype=np.float64) / (size * size)


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


def high_pass(real, imag, cutoff):
    rows, cols = real.shape
    i = np.arange(rows)[:, None] - rows // 2
    j = np.arange(cols)[None, :] - cols // 2
    mask = np.sqrt(i ** 2 + j ** 2) > cutoff
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
    real, imag = high_pass(real, imag, cutoff)
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


def to_data_uri(arr):
    """Encode a uint8 image array as a base64 PNG so it can go straight into <img>."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blur", methods=["POST"])
def blur():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")

    ksize = int(request.form.get("ksize", 5))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="That file could not be read as an image.")

    img.thumbnail((MAX_DIM, MAX_DIM))
    original = np.asarray(img, dtype=np.uint8)

    kernel = box_kernel(ksize)
    source = original.astype(np.float64)

    start = time.time()
    blurred = np.zeros_like(source)
    for c in range(3):  # convolve each colour channel independently
        blurred[:, :, c] = convolve2d(source[:, :, c], kernel)
    elapsed = time.time() - start

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        blurred=to_data_uri(blurred),
        ksize=ksize,
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
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

    return render_template(
        "index.html",
        original=to_data_uri(original),
        sharpened=to_data_uri(sharpened),
        cutoff="{:g}".format(cutoff),
        amount="{:g}".format(amount),
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
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

    return render_template(
        "index.html",
        original=to_data_uri(original),
        edges=to_data_uri(detected),
        edge_cutoff="{:g}".format(cutoff),
        size="{} x {}".format(original.shape[1], original.shape[0]),
        elapsed="{:.2f}".format(elapsed),
    )


if __name__ == "__main__":
    app.run(debug=True)
