"""Encrypt: hide the picture in the frequency domain.

Everything else here changes *what* a spectrum holds -- denoise cuts bins out,
compress throws the small ones away. This changes only *where* the energy sits.
A mask of the form e^(i*phi) has magnitude 1, so multiplying a spectrum by one
removes nothing and adds nothing: it only re-phases. Yet phase is where a
picture keeps its structure, so scrambling it turns the image into noise -- and
because nothing was discarded, the right key puts every coefficient back
exactly where it started.

This page only goes one way. Decrypt is its own page, and the way to see the
round trip is the way you would really use it: encrypt here, press Save png,
then open that file on Decrypt with the passphrase.

Two ciphers, both keyed by a passphrase:

    drpe    Double random phase encoding (Refregier & Javidi, 1995), the
            classical optical cipher. One random phase mask multiplies the
            image, a second multiplies its spectrum:

                E = F^-1 { F{ x . R1 } . R2 }

            Both masks are needed to invert it. The first is what whitens the
            spectrum: without it, F{x} still has a photo's magnitude and R2
            leaves magnitudes alone. The cost is that E is complex, and an
            image has no room for that, so the saved picture is two images
            stacked: the real part above the imaginary part, twice as tall
            as the field and twice the bytes (common/cipher_file.py). With
            both halves in the file, Decrypt opens it with the key.

    phase   One random phase mask in the spectrum, built conjugate-symmetric
            so the inverse comes back real:

                E = F^-1 { F{x} . e^(i*phi) },   phi[-u,-v] = -phi[u,v]

            Now the ciphertext is a real image. It is stretched to 0..255 and
            saved as an ordinary PNG, which is a file you can keep and open
            again later with nothing but the key. The honest weakness comes
            with it: a phase-only mask leaves |F{x}| untouched, so the
            cipher's spectrum is the plaintext's, bin for bin -- the two
            spectrum plates are identical, and that is a real leak, not a
            drawing artefact.

The scheme is not a modern cipher. DRPE is linear, and a linear cipher falls to
a known-plaintext attack; the passphrase is hashed once with sha256, which is
not a key derivation function. It is here because it is the Fourier transform
used as a cipher, which is worth seeing.

Where the code is
    common/phase_keys.py   the passphrase -> the two masks
    common/cipher_file.py  the stretch, and the tags written into the PNG
    features/encrypt.py    this file: encryption, and the /encrypt page
    features/decrypt.py    decryption, and the /decrypt page

    The two feature files do not import each other. Everything they share is
    in the two common modules above, which is what keeps them in step.

How it works
    1. masks          the passphrase -> two unit-magnitude masks
    2. encrypt        fft2 each channel, mask, ifft2  -> the cipher field
    3. visible        the field stretched into an 8-bit picture
    4. tags           the cipher, the stretch and the crop, written into the PNG

    The 2D FFT comes from common/fourier_2d.py, built on transforms.py.

Compared against
    The same encryption run on numpy.fft. Same algorithm, so the numbers mean
    something: they measure transforms.py itself, as on the denoise page.
"""

import numpy as np
from flask import render_template, request

from backend.common.cipher_file import tags, visible
from backend.common.fourier_2d import fft2, ifft2, luma, pad_to_pow2, shift
from backend.common.limits import WORK_DIM
from backend.common.metrics import compare
from backend.common.phase_keys import masks
from backend.common.spectrum import spectrum_plate
from backend.common.timing import timed, timing_fields
from backend.common.uploads import open_image, size_text, to_data_uri, uploaded_file


# what the form falls back to when the passphrase box is left empty
DEFAULT_KEY = "cse220"


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def encrypt(rgb, key, scheme):
    """Encrypt a float RGB image in 0..1.

    Returns the cipher field (complex, at the padded size), the magnitude of
    the masked luma spectrum, and the original (height, width) so a later
    decryption can crop the padding off.

    The padding cannot be cropped off the ciphertext the way it is elsewhere:
    every output pixel is a sum over every input pixel, so throwing part of the
    field away would take part of every pixel with it.
    """
    padded, crop = pad_to_pow2(rgb)
    spatial, frequency = masks(key, scheme, padded.shape[:2])

    cipher = np.zeros(padded.shape, dtype=np.complex128)
    masked = []
    for c in range(3):
        # the image-side mask goes on before the transform, the spectrum-side
        # one after it. For "phase" the first is all ones, and only the second
        # does any work.
        spectrum = fft2(padded[:, :, c] * spatial) * frequency
        masked.append(spectrum)
        cipher[:, :, c] = ifft2(spectrum)

    # the transform is linear and all three channels share the masks, so the
    # luma's masked spectrum is the same weighted sum of theirs (as in compress)
    mag = np.abs(masked[0] * 0.299 + masked[1] * 0.587 + masked[2] * 0.114)
    return cipher, mag, crop


# ---------------------------------------------------------------------------
# How well it hides: the two numbers this kind of cipher is judged on
# ---------------------------------------------------------------------------

def adjacent_correlation(gray):
    """How much each pixel is worth to its right-hand neighbour, as a correlation.

    A photograph is smooth, so this sits near 0.95: knowing one pixel nearly
    tells you the next. Whitened noise has no such link and lands near 0, and
    that drop is the thing a cipher like this is asked to show.
    """
    left = gray[:, :-1].ravel()
    right = gray[:, 1:].ravel()
    if left.size == 0 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def entropy(u8):
    """Shannon entropy of the 8-bit values, in bits per value.

    8.0 is a perfectly flat histogram -- every value equally likely, nothing to
    guess from the counts alone. A photograph is usually around 7.
    """
    counts = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def numpy_encrypt(rgb, key, scheme):
    """The same encryption, with numpy's FFT in place of the hand-written one.

    Returns the same thing encrypt() does, minus the spectrum, so the page can
    compare the two ciphertexts picture for picture.

    NOTE: numpy.fft appears three times in the project -- here, in decrypt.py
    and in denoise.py -- every time in a library baseline rather than in
    transforms.py, and every time purely as a check on the hand-written
    transform. Delete this function and the `library=` argument below if the
    assignment forbids the import anywhere in the tree.
    """
    padded, _ = pad_to_pow2(rgb)
    spatial, frequency = masks(key, scheme, padded.shape[:2])

    cipher = np.zeros(padded.shape, dtype=np.complex128)
    for c in range(3):
        cipher[:, :, c] = np.fft.ifft2(np.fft.fft2(padded[:, :, c] * spatial) * frequency)
    return cipher


# ---------------------------------------------------------------------------
# Web route: POST /encrypt
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    scheme = request.form.get("scheme", "drpe")
    key = request.form.get("key", "").strip() or DEFAULT_KEY

    # WORK_DIM, the same cap as denoise and compress (see limits.py for why):
    # a run here is two 2D transforms per channel
    _, original = open_image(file, WORK_DIM)   # numpy_encrypt needs no Pillow image
    clean = original.astype(np.float64) / 255.0

    (cipher, mag, crop), elapsed = timed(encrypt, clean, key, scheme)
    library, elapsed_lib = timed(numpy_encrypt, clean, key, scheme)

    # the field as an 8-bit picture, and the range it was stretched from
    picture, span = visible(cipher, scheme)
    picture_u8 = np.round(picture * 255.0).astype(np.uint8)

    library_picture, _ = visible(library, scheme)
    library_u8 = np.round(library_picture * 255.0).astype(np.uint8)

    # the plaintext's own spectrum, drawn beside the ciphertext's. For "phase"
    # the two are the same picture, which is the point being made.
    padded, _ = pad_to_pow2(clean)
    mag_plain = np.abs(fft2(luma(padded)))

    # The plaintext figures are taken on the padded image, because that is the
    # grid the ciphertext covers, so both columns of the stats table are
    # measured the same way. The phase cipher then reads as it should: its
    # correlation barely moves, since a unit-magnitude mask leaves the power
    # spectrum, and so the autocorrelation, exactly where it found them. The
    # last digit still wanders, because this measure is not circular and the
    # two edge columns it drops are not the ones the transform sees.
    plain_u8 = np.round(padded * 255.0).astype(np.uint8)

    # spectrum_plate draws a mask over the spectrum; there is no mask here, so
    # it is handed one that keeps everything and asked for the spectrum alone
    flat = np.ones_like(mag)

    # one byte a channel, as saved: for DRPE that is both halves, real and
    # imaginary, so twice the phase cipher's
    cipher_bytes = picture_u8.size

    return render_template(
        "index.html",
        original=to_data_uri(original),
        # the ciphertext plate is stamped with what a later decryption needs:
        # the cipher, the stretch, the crop. Saved with the file, so the
        # Decrypt page can open it with nothing but the passphrase.
        encrypted=to_data_uri(picture_u8, tags(scheme, span, crop)),
        spectrum=to_data_uri(spectrum_plate(shift(mag), flat, "spectrum")),
        spectrum_plain=to_data_uri(spectrum_plate(shift(mag_plain), flat, "spectrum")),
        library=to_data_uri(library_u8),
        lib_call="numpy.fft.fft2 / ifft2",
        stats=compare(picture_u8, library_u8),
        scheme=scheme,
        key=key,
        real_cipher=(scheme == "phase"),
        corr_plain="{:.3f}".format(adjacent_correlation(luma(padded))),
        corr_cipher="{:.3f}".format(adjacent_correlation(luma(picture))),
        entropy_plain="{:.2f}".format(entropy(plain_u8)),
        entropy_cipher="{:.2f}".format(entropy(picture_u8)),
        cipher_bytes="{:.1f} KB".format(cipher_bytes / 1024.0),
        bytes_raw="{:.1f} KB".format(original.size / 1024.0),
        size=size_text(original),
        cipher_size=size_text(picture_u8),
        spec_size=size_text(mag),
        **timing_fields(elapsed, elapsed_lib),
    )
