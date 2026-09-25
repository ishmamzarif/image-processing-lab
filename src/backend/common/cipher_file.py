"""What a ciphertext looks like as a file, and how to read one back.

Encrypt makes a picture; Decrypt opens one. Neither page calls the other, and
this is where they meet: the container format sits here, so the encoder and the
decoder cannot drift apart, and so each feature file holds one direction of the
cipher and nothing else.

A cipher field may be any range at all, and for DRPE it is complex. Getting
it into a PNG means two things:

    visible()       lay the field out as a picture, stretched into 0..1 for 8
                    bits, and say which range it was stretched from
    from_picture()  undo that, turning the 8-bit picture back into the field a
                    decryption can work on

A phase ciphertext is real, so the picture is simply the field. A DRPE one is
complex, and an image has no room for that, so its picture is two images
stacked: the real part on top and the imaginary part below it, twice as tall
as the field. Nothing is dropped, and so both ciphers can be decrypted.

and then the range has to travel with the picture, or the second step cannot be
done. So does the cipher that made it, and the size to crop back to. All three
are written into the PNG itself as tEXt chunks (uploads.to_data_uri), which are
part of the file and survive a save:

    tags()          what the encoder writes in
    read_tags()     what the decoder reads out, with fallbacks

None of it is secret and none of it is the key. It is the envelope, not the
letter: anyone holding the file can read that it is a phase ciphertext of a
188x256 picture stretched from -0.183 to 1.214, and still have no way to open
it without the passphrase.
"""

import numpy as np


# The tEXt keywords. Namespaced, so nothing else's metadata can be mistaken
# for ours, and a file that lacks them is recognised as not one of ours.
TAG_SCHEME = "ipl-cipher"
TAG_LOW = "ipl-low"
TAG_HIGH = "ipl-high"
TAG_HEIGHT = "ipl-height"
TAG_WIDTH = "ipl-width"


def visible(cipher, scheme):
    """The ciphertext laid out as a picture, stretched to 0..1, with the range used.

    "phase" has a real ciphertext, so the picture is the ciphertext itself.

    "drpe" has a complex one, so the picture is its real part above its
    imaginary part: the whole of the field, in two halves, so from_picture()
    can put it back together. The halves share one range, which is also what
    keeps the two parts on the same scale when they are joined again.

    Either way the range is the two numbers from_picture() needs to read the
    field back out of the 8-bit values. It is one range for all three channels,
    not one each, so the colour balance of what comes back is the colour
    balance that went in.
    """
    if scheme == "phase":
        field = np.real(cipher)
    else:
        field = np.concatenate([np.real(cipher), np.imag(cipher)], axis=0)
    low, high = float(field.min()), float(field.max())
    if high <= low:
        return np.zeros_like(field), (low, high)
    return (field - low) / (high - low), (low, high)


def from_picture(picture, span, scheme):
    """The ciphertext read back out of its 8-bit picture: the stretch undone, and
    for DRPE the two halves joined again into one complex field.

    The inverse of visible(), and the one lossy step in the whole round trip:
    the values were rounded to 256 levels when the file was written, and that
    rounding is what stops a recovery from being exact.
    """
    low, high = span
    field = picture.astype(np.float64) / 255.0 * (high - low) + low
    if scheme == "phase":
        return field
    half = field.shape[0] // 2
    return field[:half] + 1j * field[half:]


def tags(scheme, span, crop):
    """What the encoder writes inside the ciphertext PNG, as strings."""
    low, high = span
    height, width = crop
    return {
        TAG_SCHEME: scheme,
        TAG_LOW: repr(float(low)),
        TAG_HIGH: repr(float(high)),
        TAG_HEIGHT: int(height),
        TAG_WIDTH: int(width),
    }


def read_tags(text, shape):
    """Pull (scheme, span, crop, found) back out of a PNG's text chunks.

    A picture carrying none of ours is taken to be a phase ciphertext stretched
    over the full 0..1 range and never cropped, which is the reading that
    recovers the most from a file that says nothing about itself. `found` tells
    the caller which of the two happened, because the fallback needs one more
    step afterwards (decrypt.restretch).
    """
    height, width = shape[:2]
    try:
        return (
            text[TAG_SCHEME],
            (float(text[TAG_LOW]), float(text[TAG_HIGH])),
            (int(text[TAG_HEIGHT]), int(text[TAG_WIDTH])),
            True,
        )
    except (KeyError, ValueError):
        return "phase", (0.0, 1.0), (height, width), False
