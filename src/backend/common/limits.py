"""Size caps. The hand-written algorithms are plain Python loops, so every pixel
has a real cost, and these keep a run down to seconds rather than minutes."""

# Blur, sharpen, edges, brightness and channels shrink the image to fit in this
# many pixels on its longer side. The blur is the reason: its convolution is a
# literal per-pixel Python loop, so cost grows with the pixel count times the
# kernel area. At 500px a 3x3 blur takes ~0.9s and an 11x11 blur takes ~10s,
# which is why the kernel slider in the template stops at 11.
#
# The front end needs the same number: the template sizes the pre-run preview
# with it, and conv-viz.js / adjust.js reproduce the same downscale in the
# browser. app.py hands it to the template, so change it here and nowhere else.
MAX_DIM = 500

# Denoise and compress work at this size instead. Their transform is a
# Python-level butterfly loop, so 256x256 is about 1.5 s for three channels and
# 512x512 about eight times that. The spectrum is also easier to read when it
# is not enormous. And it keeps the padded spectrum within 65536 bins, which is
# what lets compress.py store coefficient positions as uint16.
WORK_DIM = 256
