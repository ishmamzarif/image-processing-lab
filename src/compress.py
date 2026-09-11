"""Image compression by keeping only the strongest Fourier coefficients.

The same observation denoise.py rests on, used the other way round: a photo's
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

All the transforms come from denoise.fft2 / ifft2, which are built on
transforms.FFTTransformer. numpy does array arithmetic and a partial sort, and
nothing else.
"""

import numpy as np

import denoise


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
    the web view draws as the spectrum plate.
    """
    padded, (h, w) = denoise.pad_to_pow2(rgb)
    m, n = padded.shape[:2]
    size = m * n

    spectra = [denoise.fft2(padded[:, :, c]) for c in range(3)]

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
        out[:, :, c] = np.real(denoise.ifft2(flat.reshape(m, n)))[:h, :w]

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
