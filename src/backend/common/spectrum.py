"""Drawing a spectrum as a picture, for the denoise and compress pages, and as a
height field for the 3D view on the denoise page."""

import numpy as np


def spectrum_picture(mag):
    """log(1 + |X|), scaled to 0..1.

    Without the log this is a single white dot on black: the centre bin is
    orders of magnitude above everything else.
    """
    s = np.log1p(mag)
    top = s.max()
    return s / top if top > 0 else s


def spectrum_plate(mag, mask, view):
    """Draw a centred magnitude spectrum as a uint8 RGB image, with a mask shown over it.

    `mask` is 1 where a frequency is kept and 0 where it is thrown away.
    `view` picks what is drawn:

        "spectrum"   the spectrum alone
        "removed"    only the part the mask throws away
        otherwise    the whole spectrum, with the thrown-away part tinted blue
    """
    s = spectrum_picture(mag)

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


def spectrum_surface(mag, mask, size=64):
    """A centred spectrum and its mask as a height field, for the 3D view on the denoise page.

    Returned as plain lists, ready for tojson: "before" and "after" are heights
    in 0..1 (the spectrum as given, and after the mask), "mask" is the mask.

    Pooled down to at most size x size so the browser can turn it smoothly.
    Heights are pooled by their maximum, so a noise peak one bin wide survives
    as a spike instead of being averaged into its neighbours, and the mask by
    its minimum, so a notch still shows as a hole.

    Heights are log(1 + |X|), stretched so the lowest bin of the spectrum as
    given is 0 and its highest is 1. The masked spectrum is put on the same
    scale, so the two views compare directly, and whatever the mask removed
    falls to the floor.
    """
    h, w = mag.shape
    bh, bw = max(1, h // size), max(1, w // size)

    def pool(a, how):
        # the sides are powers of two, so the blocks divide them exactly
        return how(a.reshape(h // bh, bh, w // bw, bw), axis=(1, 3))

    before = np.log1p(pool(mag, np.max))
    after = np.log1p(pool(mag * mask, np.max))
    low, top = float(before.min()), float(before.max())
    span = top - low if top > low else 1.0

    def heights(s):
        return np.round(np.clip((s - low) / span, 0.0, 1.0), 3).tolist()

    return {
        "before": heights(before),
        "after": heights(after),
        "mask": np.round(pool(mask, np.min), 2).tolist(),
    }
