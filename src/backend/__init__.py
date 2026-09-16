"""Everything that runs on the server.

    transforms.py   the course's hand-written Fourier transforms (FFT, DFT, NTT)
    common/         helpers used by more than one feature
    features/       one file per operation on the page: blur, sharpen, edges,
                    denoise, compress, brightness, channels

Each feature file has the same three parts, in the same order: the hand-written
algorithm, the library version it is compared against, and the web route that
runs both and renders the page.
"""
