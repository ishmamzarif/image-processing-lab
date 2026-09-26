"""Compress: keep only the strongest Fourier coefficients, throw the rest away.

The same observation denoise rests on, used the other way round: a photo's
energy is packed into a small part of its spectrum. Smooth regions are low
frequencies near the centre, and edges and texture spread thinly outwards. So
if we keep the few percent of coefficients with the largest magnitude and zero
everything else, the inverse transform still looks like the picture -- it just
loses fine texture first, and at very low settings starts to ring.

"Compressed" has to mean fewer bytes, not just fewer numbers, so the storage is
counted honestly. Each kept coefficient costs its position plus its value, and
that overhead is the whole difficulty: at float64 a coefficient is ~20 bytes
against 1 byte per raw pixel, so keeping even 10% would make the file bigger.
Three things bring it down here:

  * one shared set of positions for all three channels, chosen on the luma
    spectrum (the same trick denoise uses to find peaks), so each position is
    stored once rather than three times;
  * conjugate symmetry: a real image has X[-u,-v] = conj X[u,v], so only one
    bin of each pair is stored and the decoder rebuilds the other;
  * float16 values and uint16 positions.

That comes to 14 bytes per stored pair, about 7 per kept bin, against 3 per
raw RGB pixel -- roughly 3 / (7 * keep) times smaller on a power-of-two image,
so 5% kept is ~8.6x and break-even is near 43%. Real codecs go much further by
quantising coarsely and entropy-coding what is left (which is what JPEG does,
and what the JPEG-lite codec will do); this is the plain version of the idea.

How it works
    encode   pad, fft2 each channel, pick the top `keep` bins on luma, pack them
    decode   unpack, rebuild each bin's conjugate partner, ifft2 each channel

    The 2D FFT comes from common/fourier_2d.py, built on transforms.py. numpy
    does array arithmetic and a partial sort, and nothing else.

Compared against
    Pillow's JPEG, with its quality searched until the file is about as many
    bytes as ours. A different method, so the page compares size, loss and PSNR
    rather than pixels.
"""

import io

import numpy as np
from flask import render_template, request
from PIL import Image

from backend.common.fourier_2d import fft2, ifft2, pad_to_pow2, shift
from backend.common.limits import WORK_DIM
from backend.common.metrics import pixel_loss, psnr
from backend.common.spectrum import spectrum_plate, spectrum_surface
from backend.common.timing import timed, timing_fields
from backend.common.uploads import clamp, open_image, size_text, to_data_uri, uploaded_file


# ---------------------------------------------------------------------------
# Hand-written algorithm
# ---------------------------------------------------------------------------

def conjugate_index(shape):
    """Flat index of every bin's conjugate partner.

    Bin (u, v) pairs with (-u mod M, -v mod N). For a real image the two hold
    complex conjugates, so they carry the same information. A few bins (DC, and
    the Nyquist rows and columns) are their own partner, and those are real.
    """
    m, n = shape
    u = (-np.arange(m)) % m
    v = (-np.arange(n)) % n
    return (u[:, None] * n + v[None, :]).ravel()


def encode(rgb, keep):
    """Compress a float RGB image in 0..1, keeping a `keep` fraction of the spectrum.

    Returns the packed form and the luma magnitude spectrum (unshifted), which
    the web view draws as the spectrum plate. The packed form is a dict:

        shape    (M, N) of the padded spectrum
        crop     (h, w) of the original image, to cut the padding off again
        idx      flat positions of the stored bins, uint16
        values   float16 array (3 channels, stored bins, [real, imag])
    """
    padded, (h, w) = pad_to_pow2(rgb)
    m, n = padded.shape[:2]
    size = m * n

    spectra = [fft2(padded[:, :, c]) for c in range(3)]

    # The transform is linear, so the spectrum of the luma is the same weighted
    # sum of the channel spectra -- no fourth transform needed.
    luma = spectra[0] * 0.299 + spectra[1] * 0.587 + spectra[2] * 0.114
    mag = np.abs(luma)

    count = int(np.clip(round(keep * size), 1, size))
    top = np.argpartition(mag.ravel(), -count)[-count:]

    # A pair has equal magnitudes, so the partial sort nearly always takes both
    # halves already; at the cut-off it can split one, and this adds the other.
    partner = conjugate_index((m, n))
    chosen = np.zeros(size, dtype=bool)
    chosen[top] = True
    chosen |= chosen[partner]

    # one bin per pair is enough: the one with the smaller flat index
    idx = np.flatnonzero(chosen & (np.arange(size) <= partner))

    # Divided by M*N before going to float16. Unscaled, the DC bin of a white
    # 256x256 channel is 65536, which is past float16's largest value (65504).
    values = np.empty((3, idx.size, 2), dtype=np.float16)
    for c in range(3):
        kept = spectra[c].ravel()[idx] / size
        values[c, :, 0] = kept.real
        values[c, :, 1] = kept.imag

    packed = {
        "shape": (m, n),
        "crop": (h, w),
        # uint16 reaches 65535, which is exactly enough for 256x256 (WORK_DIM)
        "idx": idx.astype(np.uint16 if size <= 65536 else np.uint32),
        "values": values,
    }
    return packed, mag


def decode(packed):
    """Put the kept coefficients back, rebuild their partners, and invert."""
    m, n = packed["shape"]
    h, w = packed["crop"]
    size = m * n

    idx = packed["idx"].astype(np.int64)
    partner = conjugate_index((m, n))[idx]

    out = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        vals = packed["values"][c].astype(np.float64)
        kept = (vals[:, 0] + 1j * vals[:, 1]) * size

        flat = np.zeros(size, dtype=np.complex128)
        flat[partner] = np.conj(kept)
        flat[idx] = kept           # self-partnered bins: keep the stored value

        # with both halves of every pair restored the result is real up to
        # round-off, so dropping the imaginary part loses nothing
        out[:, :, c] = np.real(ifft2(flat.reshape(m, n)))[:h, :w]

    return np.clip(out, 0.0, 1.0)


def packed_bytes(packed):
    """What the compressed form would take on disk: positions plus values."""
    return int(packed["idx"].nbytes + packed["values"].nbytes)


def kept_mask(packed):
    """1 where a coefficient survived (either half of a pair), 0 elsewhere."""
    m, n = packed["shape"]
    idx = packed["idx"].astype(np.int64)
    mask = np.zeros(m * n, dtype=np.float64)
    mask[idx] = 1.0
    mask[conjugate_index((m, n))[idx]] = 1.0
    return mask.reshape(m, n)


def detail_energy_kept(mag, mask):
    """How much of the luma's energy the kept bins hold, as a percentage.

    Parseval: a picture's energy is the same summed over its pixels or over its
    spectrum, so this can be counted bin by bin. The zero-frequency bin, the
    average brightness, is left out of both sums. It is always kept and holds
    most of the energy on its own (72% on the cat photo), so counting it would
    make any setting look like 99%.
    """
    energy = mag ** 2
    energy[0, 0] = 0.0                 # unshifted: zero frequency is the corner
    total = float(energy.sum())
    return "{:.1f}".format(100.0 * float((energy * mask).sum()) / total if total > 0 else 100.0)


def kilobytes(n):
    return "{:.1f} KB".format(n / 1024.0)


# ---------------------------------------------------------------------------
# Library baseline
# ---------------------------------------------------------------------------

def library_jpeg(img, target_bytes):
    """Pillow's JPEG, with the quality searched until the file is about as big as ours.

    Returns the decoded JPEG pixels, its size in bytes, and the quality used.

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


# ---------------------------------------------------------------------------
# Web route: POST /compress
# ---------------------------------------------------------------------------

def view():
    file = uploaded_file()

    # capped at 25%, the same as the slider. A kept bin costs about 7 bytes
    # against 3 per raw pixel, so 25% is only ~1.7x smaller and break-even is
    # near 43%; past the cap the "compressed" file stops earning the name.
    keep = clamp(float(request.form.get("keep", 5)), 0.5, 25.0)

    # WORK_DIM, the same cap as denoise (see limits.py for why)
    img, original = open_image(file, WORK_DIM)
    clean = original.astype(np.float64) / 255.0

    (packed, mag), elapsed_enc = timed(encode, clean, keep / 100.0)
    restored, elapsed_dec = timed(decode, packed)

    # rounded to uint8 before measuring, so both PSNR figures are taken on the
    # 8-bit images actually shown, and neither side gets a precision advantage
    restored = np.round(restored * 255.0).astype(np.uint8)

    bytes_raw = original.size
    bytes_ours = packed_bytes(packed)
    mask = kept_mask(packed)

    (library, bytes_lib, jpeg_q), elapsed_lib = timed(library_jpeg, img, bytes_ours)

    elapsed = elapsed_enc + elapsed_dec

    loss_ours = pixel_loss(original, restored)
    loss_lib = pixel_loss(original, library)

    return render_template(
        "index.html",
        original=to_data_uri(original),
        compressed=to_data_uri(restored),
        spectrum=to_data_uri(spectrum_plate(shift(mag), shift(mask), "mask")),
        # the same spectrum in 3D (surface-3d.js), with the kept bins tinted:
        # as given, and with everything else zeroed, which is all the decoder
        # gets
        surface=spectrum_surface(shift(mag), shift(mask), shift(mask) > 0.5, ("Spectrum", "Kept")),
        energy=detail_energy_kept(mag, mask),
        library=to_data_uri(library),
        lib_call="Image.save(format=\"JPEG\", quality={})".format(jpeg_q),
        different_method=True,            # not the same algorithm: compared by PSNR at equal size
        keep="{:g}".format(keep),
        kept="{:.1f}".format(100.0 * float(np.mean(mask))),
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
        size=size_text(original),
        spec_size=size_text(mag),
        **timing_fields(elapsed, elapsed_lib),
    )
