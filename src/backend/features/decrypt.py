"""Decrypt: take an encrypted picture and a passphrase, and get the photo back.

The other half of Encrypt, and its own page. Encrypt shows the whole round trip
at once, which proves the maths but hides the interesting part: that the
ciphertext is a file you can keep, carry, and open again later with nothing but
a passphrase. That is what this page is. You run Encrypt, press Save png, pick
the saved file here, and type the key.

How it undoes what encrypt.py did
    Encryption was

        E = F^-1 { F{ x . R1 } . R2 }

    and every step of it is a multiplication by something of magnitude 1 or a
    transform, so every step has an inverse. Conjugating a unit-magnitude mask
    inverts it, so the same two masks, conjugated and applied in the opposite
    order, put every coefficient back where it started:

        x = conj(R1) . F^-1 { F{E} . conj(R2) }

    Nothing was discarded on the way in -- no bin was zeroed, no magnitude was
    rounded -- so this is exact, not approximate. The error is float round-off,
    about 10^-16 of the signal, and what the page shows is the 8-bit file's
    quantisation on top of that.

What the file has to carry
    A passphrase draws the masks, but three other things are needed to read a
    ciphertext, and none of them is secret: which cipher made it, the stretch
    that fitted it into 0..255, and the size to crop back to. Encrypt writes
    all three into the PNG as tEXt chunks, so they travel inside the picture
    and the user carries only the key. The format of that envelope is in
    common/cipher_file.py, which both pages read it from -- neither feature
    file imports the other.

    A file without them still decrypts. The whole pipeline is linear, and a
    constant image comes back as itself, so decrypting the stretched picture
    instead of the ciphertext gives an affine of the photo -- the right picture
    at the wrong brightness and contrast. Stretching that back out recovers it.
    What cannot be recovered is the crop, so such a result keeps the reflected
    padding it was encrypted with.

Both ciphers
    A phase ciphertext is real, and its file is simply the field. A DRPE one is
    complex, so Encrypt saves its real part above its imaginary part, one
    picture twice as tall as the field, and cipher_file.from_picture() joins the
    two halves again. With both halves nothing is missing, and it decrypts as
    exactly as the phase cipher does; the 8-bit rounding just lands on two
    numbers per pixel instead of one, so it comes back a little noisier.

With the wrong passphrase
    The masks do not cancel. conj(R') against R leaves e^(i(phi - phi')), which
    is another random phase, so what comes out is another scrambling rather
    than a damaged photo. There is no partial credit for a near-miss key: the
    passphrase is hashed before the masks are drawn, so one character's
    difference is a whole different key.

Where the code is
    common/phase_keys.py   the passphrase -> the two masks
    common/cipher_file.py  the stretch, and the tags read out of the PNG
    features/encrypt.py    encryption, and the /encrypt page
    features/decrypt.py    this file: decryption, and the /decrypt page

Compared against
    The same decryption run on numpy.fft, as on the Encrypt and Denoise pages.
"""

import numpy as np
from flask import render_template, request

from backend.common.cipher_file import from_picture, read_tags
from backend.common.fourier_2d import fft2, ifft2
from backend.common.metrics import compare
from backend.common.phase_keys import masks
from backend.common.timing import timed, timing_fields
from backend.common.uploads import UploadError, open_image_as_is, size_text, to_data_uri, uploaded_file
from backend.transforms import next_power_of_two


# what the form falls back to when the passphrase box is left empty
DEFAULT_KEY = "cse220"


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def recover(cipher, key, scheme):
    """The field the key pulls back out of the ciphertext, still complex.

    Four transforms and two conjugated masks, per colour channel. Whether the
    key was right is not knowable here and nothing in here depends on it: the
    same arithmetic runs either way, and it is plaintext() that turns the
    result into pixels.
    """
    spatial, frequency = masks(key, scheme, cipher.shape[:2])

    out = np.zeros(cipher.shape, dtype=np.complex128)
    for c in range(3):
        out[:, :, c] = ifft2(fft2(cipher[:, :, c]) * np.conj(frequency)) * np.conj(spatial)
    return out


def plaintext(field, scheme):
    """The recovered field as pixels: the picture a decryption hands back.

    Which part of a complex field is the picture depends on the cipher:

        phase   the scheme is built to keep everything real, so the real part
                is the picture and the imaginary part is round-off.
        drpe    the field is complex by design, so the magnitude is. This is
                how the DRPE literature reads its output too.

    With the right key the two rules agree exactly, because the recovered field
    is then the original image: real, and never negative. They part company
    only when the key is wrong, and there the magnitude is the honest picture.
    Taking the real part of a failed DRPE decryption would give something
    zero-mean, and clipping that to 0..1 would floor half of every such plate
    to black -- a picture of the clip, not of what the wrong key produced.
    """
    pixels = np.real(field) if scheme == "phase" else np.abs(field)
    return np.clip(pixels, 0.0, 1.0)


def decrypt(cipher, key, scheme, crop):
    """Undo encrypt.encrypt() with a passphrase, and crop the padding off.

    The crop happens here rather than on the ciphertext because it cannot
    happen any earlier: every pixel of E is a sum over every padded pixel, so
    trimming the field would take a piece of every pixel with it.
    """
    height, width = crop
    return plaintext(recover(cipher, key, scheme), scheme)[:height, :width]


def restretch(image):
    """Pull an image back out to the full 0..1 range.

    Only used when the file carried no stretch of its own. Decrypting the
    stretched picture rather than the ciphertext gives an affine of the photo,
    and this undoes that affine the only way it can be undone without being
    told what it was: by assuming the picture uses the whole range, which a
    photograph very nearly does.
    """
    low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image)
    return (image - low) / (high - low)


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def numpy_decrypt(cipher, key, scheme, crop):
    """The same decryption, with numpy's FFT in place of the hand-written one.

    NOTE: numpy.fft appears three times in the project -- here, in encrypt.py
    and in denoise.py -- every time in a library baseline rather than in
    transforms.py, and every time purely as a check on the hand-written
    transform. Delete this function and the `library=` argument below if the
    assignment forbids the import anywhere in the tree.
    """
    height, width = crop
    spatial, frequency = masks(key, scheme, cipher.shape[:2])

    out = np.zeros(cipher.shape, dtype=np.complex128)
    for c in range(3):
        spectrum = np.fft.fft2(cipher[:, :, c]) * np.conj(frequency)
        out[:, :, c] = np.fft.ifft2(spectrum) * np.conj(spatial)

    return plaintext(out, scheme)[:height, :width]


# ---------------------------------------------------------------------------
# Web route: POST /decrypt
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    # its own box, not the one Encrypt fills in: the passphrase typed here is a
    # guess at the file's key, and nothing on this page knows whether it is
    # right. A wrong one decrypts just as far and gives noise.
    key = request.form.get("guess", "").strip() or DEFAULT_KEY

    # opened at its own size: a ciphertext cannot be resampled, and the tEXt
    # chunks Encrypt left in it are half of what this page needs
    ciphertext, text = open_image_as_is(file)
    scheme, span, crop, tagged = read_tags(text, ciphertext.shape)

    # The radix-2 FFT takes power-of-two sides, and a real ciphertext always
    # has them -- it was made at the padded size. So a picture that does not is
    # not one of ours, and saying that is more use than a transform error.
    height, width = ciphertext.shape[:2]
    if next_power_of_two(height) != height or next_power_of_two(width) != width:
        raise UploadError(
            "A ciphertext is a power-of-two size, and this is {} x {}. "
            "Run Encrypt, save that picture, and choose it here."
            .format(width, height)
        )

    # for DRPE this also joins the real and imaginary halves back together
    field = from_picture(ciphertext, span, scheme)

    recovered, elapsed = timed(decrypt, field, key, scheme, crop)
    library, elapsed_lib = timed(numpy_decrypt, field, key, scheme, crop)

    # with no stretch to undo, what came back is an affine of the picture, and
    # this is the only way to put it back on the right scale
    if not tagged:
        recovered = restretch(recovered)
        library = restretch(library)

    recovered_u8 = np.round(recovered * 255.0).astype(np.uint8)
    library_u8 = np.round(library * 255.0).astype(np.uint8)

    return render_template(
        "index.html",
        ciphertext=to_data_uri(ciphertext),
        recovered=to_data_uri(recovered_u8),
        library=to_data_uri(library_u8),
        lib_call="numpy.fft.fft2 / ifft2",
        stats=compare(recovered_u8, library_u8),
        key=key,
        scheme=scheme,
        tagged=tagged,
        cipher_size=size_text(ciphertext),
        size=size_text(recovered_u8),
        span_low="{:.3f}".format(span[0]),
        span_high="{:.3f}".format(span[1]),
        **timing_fields(elapsed, elapsed_lib),
    )
