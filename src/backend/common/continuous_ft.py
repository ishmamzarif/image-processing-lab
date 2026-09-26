"""A continuous Fourier transform, evaluated numerically. Used by sharpen and edges.

This is a different tool from fourier_2d.py and does not use transforms.py.
Instead of the discrete sums of an FFT, the image is treated as a function
f(x, y) sampled on [-1, 1] x [-1, 1], and the defining integrals

    F(u, v) = integral integral  f(x, y) e^(-2 pi i (u x + v y))  dx dy      (forward)
    f(x, y) = integral integral  F(u, v) e^(+2 pi i (u x + v y))  du dv      (inverse)

are approximated with the trapezoidal rule (np.trapezoid). The complex
exponential is split into cos and sin so everything stays real, and each double
integral is done one axis at a time: first along x (or v), then along y (or u).

The pipeline both features use is high_pass_detail():

    axes  ->  compute_cft  ->  high_pass (drop low frequencies)  ->  reconstruct
"""

import numpy as np


def spatial_axes(height, width):
    """Sample positions of the pixels: `width` points across [-1, 1], `height` down it."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    return x, y


def frequency_axes(x, y):
    """Frequencies to evaluate at: one per pixel, from -Nyquist to +Nyquist.

    Nyquist is 1 / (2 * sample spacing), the fastest wave the samples can hold.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), x.size)
    v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), y.size)
    return u, v


def compute_cft(channel, x, y, u, v):
    """Forward transform of one channel. Returns F's real and imaginary parts, (v, u) shaped.

    Step 1, along x, for every row y and every u:
        r1 = integral f cos(2 pi u x) dx,   r2 = integral f sin(2 pi u x) dx
    so that integral f e^(-2 pi i u x) dx = r1 - i r2.

    Step 2, along y, multiplying by e^(-2 pi i v y) = cos - i sin:
        (r1 - i r2)(cos - i sin) = (r1 cos - r2 sin) - i (r2 cos + r1 sin)
    """
    # step 1: integrate every row along x, once per horizontal frequency u
    r1 = np.zeros((y.size, u.size), dtype=np.float64)
    r2 = np.zeros((y.size, u.size), dtype=np.float64)
    for j in range(u.size):
        theta = 2 * np.pi * u[j] * x
        r1[:, j] = np.trapezoid(channel * np.cos(theta), x, axis=1)
        r2[:, j] = np.trapezoid(channel * np.sin(theta), x, axis=1)

    # step 2: integrate those down y, once per vertical frequency v
    real = np.zeros((v.size, u.size), dtype=np.float64)
    imag = np.zeros((v.size, u.size), dtype=np.float64)
    for i in range(v.size):
        theta = 2 * np.pi * v[i] * y
        cos_v_y = np.cos(theta)[:, None]
        sin_v_y = np.sin(theta)[:, None]
        real[i, :] = np.trapezoid(r1 * cos_v_y - r2 * sin_v_y, y, axis=0)
        imag[i, :] = -np.trapezoid(r2 * cos_v_y + r1 * sin_v_y, y, axis=0)

    return real, imag


def high_pass(real, imag, cutoff, u, v):
    """Drop the low frequencies, keeping what changes quickly.

    Two things here are easy to get wrong, and both were:

    1. The radius is measured off the u and v axes rather than off array
       indices. frequency_axes returns linspace(-Nyquist, +Nyquist, N), which
       for an even N has no exact zero bin -- u[N//2] sits half a bin above
       zero. Centring the mask on index N//2 therefore centres it half a bin
       off DC, which tilts the whole filter.

    2. The edge is a Gaussian, not a step. A hard circle in the frequency plane
       is a sinc in the image plane, so every edge comes back wrapped in
       ripples that spread across the entire picture -- the Gibbs phenomenon.
       That is what made the old output look nothing like an unsharp mask.
       Fading out over `cutoff` instead removes the ripples entirely.

    The mask is 1 - exp(-D^2 / 2c^2): a Gaussian low-pass subtracted from
    everything, which is precisely unsharp masking done in the frequency
    domain. `cutoff` is in bins, so it keeps the meaning it had before.
    """
    mask = high_pass_mask(cutoff, u, v)
    return real * mask, imag * mask


def high_pass_mask(cutoff, u, v):
    """The mask high_pass multiplies by, (v, u) shaped: 0 at the centre, 1 far out."""
    du = u[1] - u[0]
    dv = v[1] - v[0]
    d = np.sqrt((v[:, None] / dv) ** 2 + (u[None, :] / du) ** 2)
    return 1.0 - np.exp(-(d ** 2) / (2.0 * max(cutoff, 1e-6) ** 2))


def reconstruct(real, imag, u, v, x, y):
    """Inverse transform back to an image, (y, x) shaped. Only the real part is kept.

    Step 1, along v, for every y: multiplying F = real + i imag by
    e^(+2 pi i v y) = cos + i sin gives
        r1 = integral (real cos - imag sin) dv,   r2 = integral (real sin + imag cos) dv

    Step 2, along u, for every x: the real part of (r1 + i r2)(cos + i sin) is
        r1 cos - r2 sin
    """
    # step 1: integrate over v, once per row position y
    r1 = np.zeros((y.size, u.size), dtype=np.float64)
    r2 = np.zeros((y.size, u.size), dtype=np.float64)
    for i in range(y.size):
        theta = 2 * np.pi * v * y[i]
        cos_v_y = np.cos(theta)[:, None]
        sin_v_y = np.sin(theta)[:, None]
        r1[i, :] = np.trapezoid(real * cos_v_y - imag * sin_v_y, v, axis=0)
        r2[i, :] = np.trapezoid(real * sin_v_y + imag * cos_v_y, v, axis=0)

    # step 2: integrate over u, once per column position x
    image = np.zeros((y.size, x.size), dtype=np.float64)
    for j in range(x.size):
        theta = 2 * np.pi * x[j] * u
        image[:, j] = np.trapezoid(r1 * np.cos(theta) - r2 * np.sin(theta), u, axis=1)

    return image


def high_pass_detail(channel, cutoff):
    """The fine detail of one channel: everything above `cutoff`, back in image space.

    Sharpen adds this to the image; edges takes its size as the edge strength.

    Returns (detail, spectrum, mask). The other two are for the 3D view on
    those pages, handed back so nothing is transformed twice: the spectrum
    before the mask (complex, (v, u) shaped, centred) and the mask. The
    spectrum is divided by dx dy, which turns the integral back into the plain
    sum a DFT would give, so its logarithm reads like the app's other spectra.
    """
    x, y = spatial_axes(channel.shape[0], channel.shape[1])
    u, v = frequency_axes(x, y)
    real, imag = compute_cft(channel, x, y, u, v)
    kept_real, kept_imag = high_pass(real, imag, cutoff, u, v)
    detail = reconstruct(kept_real, kept_imag, u, v, x, y)
    spectrum = (real + 1j * imag) / ((x[1] - x[0]) * (y[1] - y[0]))
    return detail, spectrum, high_pass_mask(cutoff, u, v)
