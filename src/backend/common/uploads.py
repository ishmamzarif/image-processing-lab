"""Getting the image in from the form, and getting results back out to the page.

Every route starts the same way -- is there a file, can it be opened, shrink it
to the working size -- so that lives here. When something is wrong the helpers
raise UploadError, and app.py turns that into the page with the message shown.
That way a route reads as a straight line from input to output, with no
early-return checks cluttering the top of it.
"""

import base64
import io

import numpy as np
from flask import request
from PIL import Image, PngImagePlugin


class UploadError(Exception):
    """The upload is missing or unreadable. The message is shown on the page."""


def uploaded_file():
    """The file from the form's `image` field. Raises UploadError if none was chosen."""
    file = request.files.get("image")
    if file is None or file.filename == "":
        raise UploadError("Please choose an image file.")
    return file


def open_image(file, max_dim):
    """Open the upload as RGB and shrink it (keeping its shape) to fit max_dim.

    Returns the Pillow image, which the library baselines need, and the same
    pixels as a uint8 array of shape (height, width, 3), which ours need.

    thumbnail() only ever shrinks, so a small image is left at its own size.
    """
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        raise UploadError("That file could not be read as an image.")

    img.thumbnail((max_dim, max_dim))
    return img, np.asarray(img, dtype=np.uint8)


def form_number(name, default):
    """A number from the form, or `default` when the field was not sent."""
    return float(request.form.get(name, default))


def clamp(value, low, high):
    return min(max(value, low), high)


def to_data_uri(arr, text=None):
    """Encode a uint8 image array as a base64 PNG so it can go straight into <img src>.

    Nothing is saved on the server: the pictures travel inside the page itself.

    `text` is an optional dict written into the PNG as tEXt chunks. Those are
    part of the file, so they survive a Save png and come back when the file is
    uploaded again -- which is how Encrypt hands Decrypt everything it needs
    about a ciphertext except the passphrase. Values are written as strings.
    """
    buf = io.BytesIO()

    info = None
    if text:
        info = PngImagePlugin.PngInfo()
        for keyword, value in text.items():
            info.add_text(keyword, str(value))

    Image.fromarray(arr).save(buf, format="PNG", pnginfo=info)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def open_image_as_is(file):
    """Open the upload as RGB at its own size, with whatever tEXt chunks it carries.

    open_image() shrinks what it opens, which is right for a photograph and
    wrong for a ciphertext: resampling a ciphertext destroys it completely, and
    its size is part of the data. So Decrypt reads its upload through here
    instead, and gets the picture back untouched along with the dict of text
    the encoder left inside it.
    """
    try:
        img = Image.open(file.stream)
        img.load()                      # forces the chunks to be read
        text = dict(getattr(img, "text", {}) or {})
        return np.asarray(img.convert("RGB"), dtype=np.uint8), text
    except Exception:
        raise UploadError("That file could not be read as an image.")


def size_text(arr):
    """An image's size as the page prints it: width x height."""
    return "{} x {}".format(arr.shape[1], arr.shape[0])
