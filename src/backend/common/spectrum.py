"""Drawing a spectrum as a picture, for the denoise and compress pages, and as a
height field for the 3D views (common/surface.py has the format)."""

import numpy as np

from backend.common.surface import flags, heights, pool


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


def spectrum_surface(mag, mask, tinted, labels, size=64):
    """A centred spectrum before and after a mask, as a 3D view with two modes.

    `labels` names the two buttons, and `tinted` marks the bins drawn in the
    accent colour in both modes: denoise tints what its mask removes, compress
    what its mask keeps. See common/surface.py for the format.

    Heights are pooled by their maximum, so a noise peak one bin wide survives
    as a spike instead of being averaged into its neighbours, and the tint
    likewise, so a single tinted bin still shows.

    Heights are log(1 + |X|), stretched so the lowest bin of the spectrum as
    given is 0 and its highest is 1. The masked spectrum is put on the same
    scale, so the two views compare directly, and whatever the mask removed
    falls to the floor.
    """
    before = np.log1p(pool(mag, size, np.max))
    after = np.log1p(pool(mag * mask, size, np.max))
    low, top = float(before.min()), float(before.max())
    tint = flags(pool(tinted, size, np.max))
    return {
        "axes": ["u", "v"],
        "modes": [
            {"label": labels[0], "z": heights(before, low, top), "tint": tint},
            {"label": labels[1], "z": heights(after, low, top), "tint": tint},
        ],
    }


def filter_surface(spectrum, H, tinted, out_label, half, size=64):
    """A spectrum, a filter and their product, as a 3D view with three modes.

        Spectrum    log(1 + |F|)
        Filter H    the filter itself, from 0 to its largest value
        out_label   log(1 + |H F|), on the same scale as Spectrum

    This is the whole of "filtering is multiplication" in one plot: the first
    mode times the second is the third. Used by sharpen and edges.

    `spectrum` is complex and centred, `H` real and on the same grid, and
    `tinted` marks the same bins in all three modes, so a region can be
    followed from one to the next. Only the middle `half` bins each way are
    drawn: a high-pass with a small cutoff changes a small disc in the middle,
    and over the whole spectrum its bowl would be a dimple. Returns the
    window's half-widths (u, v) as "window", for the plate's caption.
    """
    h, w = H.shape
    rv, ru = min(half, h // 2), min(half, w // 2)
    middle = (slice(h // 2 - rv, h // 2 + rv), slice(w // 2 - ru, w // 2 + ru))
    mag, H, tint = np.abs(spectrum[middle]), H[middle], flags(pool(tinted[middle], size, np.max))

    before = np.log1p(pool(mag, size, np.max))
    after = np.log1p(pool(mag * H, size, np.max))
    low, top = float(before.min()), float(before.max())
    return {
        "axes": ["u", "v"],
        "window": [ru, rv],
        "modes": [
            {"label": "Spectrum", "z": heights(before, low, top), "tint": tint},
            {"label": "Filter H", "z": heights(pool(H, size, np.mean), 0.0, float(H.max())), "tint": tint,
             "plain": True},
            {"label": out_label, "z": heights(after, low, top), "tint": tint},
        ],
    }
