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


if __name__ == "__main__":
    app.run(debug=True)
