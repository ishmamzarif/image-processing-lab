"""Figures and numbers for the slides, made by running the app's own backend.

Run from the repository root:

    .venv/bin/python presentation/make_figures.py

Every picture in presentation/figures/ comes out of the same functions the web
app calls, so nothing on the slides is mocked up. Three photos sit next to this
script, and each feature uses the one that shows it best:

    cat_soldier.jpg   the operations grid, denoise, encrypt, point operations,
                      and the timings in the results table
    blue_flower.jpg   sharpen & edges (fine petal veins), compress (smooth shading)
    casette.jpg       blur and resize (hard black-and-white lines), and the
                      ideal-vs-Gaussian ringing backup

The numbers the slides quote (timings, PSNR, byte counts, max differences) are
printed at the end; timings depend on the machine.
"""

import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from backend.common.cipher_file import from_picture, visible            # noqa: E402
from backend.common.fourier_2d import luma, pad_to_pow2, shift           # noqa: E402
from backend.common.limits import MAX_DIM, WORK_DIM                     # noqa: E402
from backend.common.metrics import compare, psnr                        # noqa: E402
from backend.common.spectrum import spectrum_plate                      # noqa: E402
from backend.features import (                                          # noqa: E402
    blur, brightness, channels, compress, decrypt, denoise, edges, encrypt, filters, resize,
    sharpen,
)
from backend.transforms import DFTAnalyzer, FFTTransformer              # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

CAT = "cat_soldier.jpg"
FLOWER = "blue_flower.jpg"
CASSETTE = "casette.jpg"

# The cassette sits in a wide white margin; this is its bounding box plus a
# little room, so the lines fill the picture on a slide.
CASSETTE_BOX = (95, 106, 491, 372)

numbers = {}


def load(name, max_dim, box=None):
    """A photo shrunk the way open_image() does: the Pillow image and a uint8 array."""
    img = Image.open(os.path.join(HERE, name)).convert("RGB")
    if box:
        img = img.crop(box)
    img.thumbnail((max_dim, max_dim))
    return img, np.asarray(img, dtype=np.uint8)


def save(name, arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, name + ".png"))


def save_zoom(name, arr, rows, cols, size=288):
    """A crop of arr, blown up with nearest-neighbour so each pixel stays square.
    `size` is one side for a square crop, or (width, height)."""
    if isinstance(size, int):
        size = (size, size)
    Image.fromarray(np.ascontiguousarray(arr[rows, cols])).resize(
        size, Image.Resampling.NEAREST).save(os.path.join(OUT, name + ".png"))


def timed(fn, *args):
    t = time.time()
    out = fn(*args)
    return out, time.time() - t


def best_of(fn, *args, runs=3):
    """Like timed(), but the fastest of a few runs: one run is noisy, and the
    results table compares times that differ by a factor of a thousand."""
    best = None
    for _ in range(runs):
        out, t = timed(fn, *args)
        best = t if best is None else min(best, t)
    return out, best


def u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def encrypt_round_trip(clean, key, wrong_key):
    """Encrypt, write the 8-bit picture a saved PNG would hold, decrypt it with two keys."""
    cipher, _, crop = encrypt.encrypt(clean, key, "phase")
    picture, span = visible(cipher, "phase")
    picture_u8 = np.round(picture * 255.0).astype(np.uint8)
    field = from_picture(picture_u8, span, "phase")   # what Decrypt reads out of the saved PNG
    right = decrypt.decrypt(field, key, "phase", crop)
    wrong = decrypt.decrypt(field, wrong_key, "phase", crop)
    return cipher, picture_u8, right, wrong


# ---------------------------------------------------------------------------
# FFT engine: hand-written vs the direct DFT and vs numpy
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
x = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
d, f = DFTAnalyzer(), FFTTransformer()
X_dft, t_dft = best_of(d.transform, x)
X_fft, t_fft = best_of(f.transform, x)
numbers["fft_vs_dft_err"] = float(np.max(np.abs(X_dft - X_fft)))
numbers["fft_vs_numpy_err"] = float(np.max(np.abs(X_fft - np.fft.fft(x))))
numbers["fft_roundtrip_err"] = float(np.max(np.abs(f.inverse(X_fft) - x)))
numbers["dft_1024_s"] = t_dft
numbers["fft_1024_s"] = t_fft

# ---------------------------------------------------------------------------
# The operations grid: every operation on the cat, small
# ---------------------------------------------------------------------------
cat_s_img, cat_s = load(CAT, 220)
cat_s_src = cat_s.astype(np.float64)
cat_s_clean = cat_s_src / 255.0
save("grid_input", cat_s)
save("grid_blur", u8(blur.blur_image(cat_s_src, blur.gaussian_kernel(9))))
save("grid_sharpen", u8(sharpen.sharpen_image(cat_s_clean, 15.0, 1.5)[0] * 255.0))
save("grid_edges", u8(edges.detect_edges(cat_s, 15.0)[0] * 255.0))
g_noisy = denoise.add_noise(cat_s_clean, "periodic", 0.3, seed=0)
g_clean, _, _, _ = denoise.denoise_image(g_noisy, "notch", 45, 3, 20)
save("grid_denoise", g_clean * 255)
g_packed, _ = compress.encode(cat_s_clean, 0.05)
save("grid_compress", np.round(compress.decode(g_packed) * 255))
_, g_cipher, g_right, _ = encrypt_round_trip(cat_s_clean, "cse220", "cse221")
save("grid_encrypt", g_cipher)
save("grid_decrypt", np.round(g_right * 255))
save("grid_brightness", u8(brightness.brighten_image(cat_s_src, 1.4)))
save("grid_channels", u8(channels.shift_image(cat_s_src, [50, 0, -50])))
save("grid_filters", u8(filters.vintage(cat_s_src, 1.0)))
save("grid_resize", resize.resize_nearest(resize.resize_nearest(cat_s_src, 0.2), 5.0))

# ---------------------------------------------------------------------------
# Blur: four kernels on the cassette, and one timed box blur on the cat at the
# app's real cap
# ---------------------------------------------------------------------------
cas_img, cas = load(CASSETTE, 240, CASSETTE_BOX)
save("blur_input", cas)
cas_src = cas.astype(np.float64)
for kind in ("box", "gaussian", "disk", "motion"):
    k = blur.make_kernel(kind, 9, 30.0)
    # the kernel itself (the impulse response) as a 9x9 grid: darker = more
    # weight, each kernel scaled to its own largest weight
    cell = 16
    kimg = np.repeat(np.repeat(255 * (1 - 0.8 * k / k.max()), cell, 0), cell, 1)
    kimg[::cell, :] = 255
    kimg[:, ::cell] = 255
    Image.fromarray(np.round(kimg).astype(np.uint8)).save(os.path.join(OUT, "kernel_" + kind + ".png"))

    ours = u8(blur.blur_image(cas_src, k))
    lib, _, _ = blur.library_blur(cas_img, cas_src, kind, 9, k)
    save("blur_" + kind, ours)
    numbers["blur_" + kind + "_maxdiff"] = compare(ours, lib)

cat_m_img, cat_m = load(CAT, MAX_DIM)
cat_m_src = cat_m.astype(np.float64)
k5 = blur.box_kernel(5)
ours, t_ours = best_of(blur.blur_image, cat_m_src, k5)
(lib, _, _), t_lib = best_of(blur.library_blur, cat_m_img, cat_m_src, "box", 5, k5)
numbers["timing_blur"] = (cat_m.shape, t_ours, t_lib, compare(u8(ours), lib))

# ---------------------------------------------------------------------------
# Sharpen and edges: continuous FT high-pass, on the flower
# ---------------------------------------------------------------------------
fl_img, fl = load(FLOWER, 256)
save("flower_input", fl)
fl_clean = fl.astype(np.float64) / 255.0
(sharp, _, _), t_sh = timed(sharpen.sharpen_image, fl_clean, 15.0, 1.5)
sharp = u8(sharp * 255.0)
(ed, _, _), t_ed = timed(edges.detect_edges, fl, 15.0)
save("edges", u8(ed * 255.0))
save("edges_pillow", edges.library_edges(fl_img))
numbers["flower_shape"] = fl.shape
numbers["sharpen_256_s"] = t_sh
numbers["edges_256_s"] = t_ed

# a close-up of the centre (stamens, petal veins), where sharpening shows at slide size
centre = (slice(72, 168), slice(80, 176))
save_zoom("sharpen_crop_before", fl, *centre)
save_zoom("sharpen_crop_after", sharp, *centre)

# ---------------------------------------------------------------------------
# Denoise: periodic stripes and the notch filter, on the cat, at the app's
# default settings (noise 0.3, notch width 3)
# ---------------------------------------------------------------------------
cat_w_img, cat_w = load(CAT, WORK_DIM)
save("cat_input", cat_w)
cat_clean = cat_w.astype(np.float64) / 255.0
noisy = denoise.add_noise(cat_clean, "periodic", 0.3, seed=0)
(cleaned, mag, mask, peaks), t_dn = best_of(denoise.denoise_image, noisy, "notch", 45, 3, 20)
lib_dn, t_dn_lib = best_of(denoise.numpy_filter, noisy, mask)
save("denoise_noisy", noisy * 255)
save("denoise_clean", cleaned * 255)

# the middle of the spectrum, magnified, so the notches read on a slide
plate = spectrum_plate(mag, mask, "mask")
cy, cx = plate.shape[0] // 2, plate.shape[1] // 2
save_zoom("denoise_mask_zoom", plate, slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))

numbers["denoise"] = {
    "peaks": len(peaks),
    "kept_percent": 100.0 * float(np.mean(mask)),
    "psnr_noisy": psnr(cat_clean, noisy),
    "psnr_clean": psnr(cat_clean, cleaned),
}
numbers["timing_denoise"] = (cat_w.shape, t_dn, t_dn_lib,
                             compare((cleaned * 255).astype(np.uint8), (lib_dn * 255).astype(np.uint8)))

# ---------------------------------------------------------------------------
# Compress: keep 5% of the flower's spectrum, against JPEG at the same size
# ---------------------------------------------------------------------------
(packed, cmag), t_enc = timed(compress.encode, fl_clean, 0.05)
restored, t_dec = timed(compress.decode, packed)
restored = np.round(restored * 255.0).astype(np.uint8)
jpeg, bytes_jpeg, q = compress.library_jpeg(fl_img, compress.packed_bytes(packed))
save("compress_ours", restored)
save("compress_jpeg", jpeg)
save("compress_mask", spectrum_plate(shift(cmag), shift(compress.kept_mask(packed)), "mask"))
numbers["compress"] = {
    "raw_bytes": fl.size,
    "ours_bytes": compress.packed_bytes(packed),
    "jpeg_bytes": bytes_jpeg,
    "jpeg_q": q,
    "psnr_ours": psnr(fl_clean, restored / 255.0),
    "psnr_jpeg": psnr(fl_clean, jpeg / 255.0),
    "enc_s": t_enc,
    "dec_s": t_dec,
}

# ---------------------------------------------------------------------------
# Encrypt / decrypt: phase-only cipher on the cat, saved as an 8-bit PNG and
# read back with the right key and a one-character-off key
# ---------------------------------------------------------------------------
(cipher, picture_u8, right, wrong), t_en = timed(encrypt_round_trip, cat_clean, "cse220", "cse221")
lib_cipher, t_en_lib = best_of(encrypt.numpy_encrypt, cat_clean, "cse220", "phase")
lib_pic, _ = visible(lib_cipher, "phase")
save("encrypt_cipher", picture_u8)
save("decrypt_right", np.round(right * 255))
save("decrypt_wrong", np.round(wrong * 255))

drpe, _, _ = encrypt.encrypt(cat_clean, "cse220", "drpe")
drpe_pic, _ = visible(drpe, "drpe")
drpe_u8 = np.round(drpe_pic * 255).astype(np.uint8)
numbers["drpe"] = {
    "corr": encrypt.adjacent_correlation(luma(drpe_pic)),
    "entropy": encrypt.entropy(drpe_u8),
}

padded, _ = pad_to_pow2(cat_clean)
numbers["encrypt"] = {
    "corr_plain": encrypt.adjacent_correlation(luma(padded)),
    "corr_cipher": encrypt.adjacent_correlation(luma(picture_u8 / 255.0)),
    "entropy_plain": encrypt.entropy(np.round(padded * 255).astype(np.uint8)),
    "entropy_cipher": encrypt.entropy(picture_u8),
    "psnr_right": psnr(cat_clean, right),
    "psnr_wrong": psnr(cat_clean, wrong),
}
# the encrypt half only, timed against the same encryption on numpy.fft
(cipher2, _, _), t_en_only = best_of(encrypt.encrypt, cat_clean, "cse220", "phase")
numbers["timing_encrypt"] = (cat_w.shape, t_en_only, t_en_lib,
                             compare(picture_u8, np.round(lib_pic * 255).astype(np.uint8)))

# ---------------------------------------------------------------------------
# Point operations and filters: a strip on the small cat, timings at the cap
# ---------------------------------------------------------------------------
save("pt_brightness", u8(brightness.brighten_image(cat_s_src, 1.4)))
save("pt_channels", u8(channels.shift_image(cat_s_src, [50, 0, -50])))
save("pt_grayscale", u8(filters.grayscale(cat_s_src, 1.0)))
save("pt_invert", u8(filters.invert(cat_s_src, 1.0)))
save("pt_bw", u8(filters.black_and_white(cat_s_src, 110)))
save("pt_pixelate", u8(filters.pixelate(cat_s_src, 10)))
save("pt_vintage", u8(filters.vintage(cat_s_src, 1.0)))

b, t_b = best_of(brightness.brighten_image, cat_m_src, 1.4)
(lb, _), t_lb = best_of(brightness.library_brightness, cat_m_img, 1.4)
numbers["timing_brightness"] = (cat_m.shape, t_b, t_lb, compare(u8(b), lb))

c, t_c = best_of(channels.shift_image, cat_m_src, [50, 0, -50])
(lc, _), t_lc = best_of(channels.library_channels, cat_m_img, [50, 0, -50])
numbers["timing_channels"] = (cat_m.shape, t_c, t_lc, compare(u8(c), lc))

g, t_g = best_of(filters.grayscale, cat_m_src, 1.0)
(lg, _), t_lg = best_of(filters.library_filter, cat_m_img, "grayscale", 1.0)
numbers["timing_grayscale"] = (cat_m.shape, t_g, t_lg, compare(u8(g), lg))

r, t_r = best_of(resize.resize_nearest, cat_m_src, 0.5)
(lr, _), t_lr = best_of(resize.library_resize, cat_m_img, 0.5)
numbers["timing_resize"] = (cat_m.shape, t_r, t_lr, compare(r.astype(np.uint8), lr))

# ---------------------------------------------------------------------------
# Resize: the cassette shrunk and blown back up, and aliasing on a zone plate
# ---------------------------------------------------------------------------
tiny = resize.resize_nearest(cas_src, 0.2)
save("resize_blocky", resize.resize_nearest(tiny, 5.0))

# A zone plate's local frequency rises from 0 at the centre to Nyquist at the
# edge. Nearest-neighbour keeps every 4th sample with no low-pass first, so
# everything above the new Nyquist folds back into false rings. Pillow's
# LANCZOS reduce low-passes first (an anti-alias filter). 192 px is about what
# the picture occupies on a projected slide, so the viewer's own resampling
# does not add a moire of its own.
n = 192
yy, xx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2].astype(np.float64)
zone = np.round(255 * (0.5 + 0.5 * np.cos(np.pi * (xx ** 2 + yy ** 2) / n))).astype(np.uint8)
zone_rgb = np.stack([zone] * 3, -1)
save("zone", zone_rgb)
zone_near = resize.resize_nearest(zone_rgb.astype(np.float64), 0.25)
Image.fromarray(zone_near.astype(np.uint8)).resize((n, n), Image.Resampling.NEAREST).save(
    os.path.join(OUT, "zone_nearest.png"))
zone_lp = Image.fromarray(zone_rgb).resize((n // 4, n // 4), Image.Resampling.LANCZOS)
zone_lp.resize((n, n), Image.Resampling.NEAREST).save(os.path.join(OUT, "zone_lowpass.png"))

# ---------------------------------------------------------------------------
# Backup: ideal vs Gaussian low-pass at the same cutoff on the cassette, where
# the hard edges make the ringing plain
# ---------------------------------------------------------------------------
_, cas_w = load(CASSETTE, WORK_DIM, CASSETTE_BOX)
cas_clean = cas_w.astype(np.float64) / 255.0
for which in ("ideal", "gaussian"):
    out, _, _, _ = denoise.denoise_image(cas_clean, which, 16, 3, 0)
    # doubled with nearest-neighbour so the ripples stay crisp on the slide
    h, w = out.shape[:2]
    save_zoom("lowpass_" + which, u8(np.round(out * 255)), slice(None), slice(None), size=(2 * w, 2 * h))

# ---------------------------------------------------------------------------
for key, value in numbers.items():
    print(key, "=", value)
