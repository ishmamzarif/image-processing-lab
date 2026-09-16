"""Drawing a spectrum as a picture, for the denoise and compress pages."""

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
