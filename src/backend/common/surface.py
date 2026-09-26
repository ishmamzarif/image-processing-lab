"""Height fields for the 3D views (static/js/surface-3d.js, drawn by plot3d.js).

A view is a plain dict, turned into JSON inside partials/surface_3d.html:

    axes    the two names drawn along the floor, e.g. ["u", "v"]
    modes   one per button on the view's toggle, in order, each with
                label   the button's text
                z       heights from 0 to 1, rows x cols, row 0 at the back
                tint    optional, rows x cols of 0 / 1: 1 where the surface
                        takes the accent colour instead of grey
                rgb     optional, rows x cols of "#rrggbb": the surface's own
                        colours, for a picture draped over its brightness
                plain   optional, true to draw in one light grey rather than
                        darker with height: for a smooth shape, such as a
                        filter, which a dark plateau would swamp

The modes of one view may have different grids. They are all stretched over
the same floor, so a coarse mode and a fine one cover the same area.

The helpers here are the parts every view shares: pooling a large array down
to a grid the browser can turn smoothly, and scaling heights to 0..1.
"""

import numpy as np


def pool(a, size, how):
    """Shrink a 2D array to at most size x size by applying `how` to each block.

    `how` is np.max, np.min or np.mean. The blocks are whole, so a side that
    does not divide evenly loses its last few rows or columns, taken equally
    off both ends: the middle stays in the middle, which for a centred
    spectrum is where zero frequency sits.
    """
    h, w = a.shape
    bh, bw = -(-h // size), -(-w // size)          # rounded up
    nh, nw = h // bh, w // bw
    top, left = (h - nh * bh) // 2, (w - nw * bw) // 2
    a = a[top:top + nh * bh, left:left + nw * bw]
    return how(a.reshape(nh, bh, nw, bw), axis=(1, 3))


def heights(a, low, top):
    """`a` stretched so `low` is 0 and `top` is 1, clipped, as nested lists.

    Several modes of one view share a `low` and `top`, so their heights can be
    compared directly: what a filter removed drops to the floor.
    """
    span = top - low if top > low else 1.0
    return np.round(np.clip((a - low) / span, 0.0, 1.0), 3).tolist()


def flags(a):
    """A boolean array as 0 / 1 lists, for a mode's `tint`."""
    return np.asarray(a, dtype=bool).astype(int).tolist()
