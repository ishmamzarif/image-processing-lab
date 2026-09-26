"""Helpers shared by more than one feature.

    limits.py         how big an image each kind of operation will work on
    uploads.py        reading the uploaded image, and turning results back into <img> sources
    timing.py         timing the two implementations and formatting the numbers for the page
    metrics.py        how close two images are (max difference, PSNR, % loss)
    fourier_2d.py     2D FFT built from transforms.py, plus padding / centring / luma
    phase_keys.py     the passphrase -> the two masks encrypt and decrypt share
    cipher_file.py    what a ciphertext looks like as a PNG, both ways
    continuous_ft.py  the continuous Fourier transform that sharpen and edges use
    spectrum.py       drawing a spectrum (and a mask over it) as a picture, or as a 3D view
    surface.py        the height fields the 3D views draw, and the helpers they share
"""
