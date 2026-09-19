"""The passphrase, and the two masks it stands for.

Both halves of the Encrypt page draw their masks here -- features/encrypt.py to
scramble with them, features/decrypt.py to undo them. Drawing them in one place
is what keeps the two halves in step: a decryption is only exact because it
rebuilds, bit for bit, the same masks the encryption used, and there is no way
for one file to change how a mask is drawn without the other following.

What a mask is
    A random phase mask is unit magnitude at every bin:

        R = e^(i*phi),      |R| = 1

    Multiplying by one adds no energy and removes none. Nothing about the
    picture is thrown away; only the phase of each coefficient turns. That is
    what makes the whole scheme invertible, and conjugating is what inverts it:

        R * conj(R) = |R|^2 = 1

Two shapes of mask, one per cipher
    phase_mask            a free random phase. The "drpe" cipher uses two of
                          these, and its ciphertext comes out complex.
    symmetric_phase_mask  a random phase whose angle is odd, which is what
                          keeps a real image real. The "phase" cipher uses one,
                          and its ciphertext is an ordinary image.
"""

import hashlib

import numpy as np


def key_stream(key, purpose, shape):
    """Uniform values in 0..1 of the given shape, drawn from the passphrase.

    sha256 turns the passphrase into 256 bits, which seed the generator. Two
    masks are needed per key and they must not be the same one twice, so
    `purpose` is hashed in with it: one passphrase gives "spatial" and
    "frequency" two unrelated streams.

    The digest is used whole. A generator seeded with all 256 bits is stirred
    just as thoroughly by a one-character change as by an entirely new
    passphrase, which is why a near-miss key gives noise rather than a blurred
    photo -- the nearness is gone before any mask is drawn.
    """
    digest = hashlib.sha256((purpose + ":" + key).encode("utf-8")).digest()
    seed = [int(word) for word in np.frombuffer(digest, dtype=np.uint32)]
    return np.random.default_rng(seed).random(shape)


def phase_mask(key, purpose, shape):
    """A random phase mask: e^(2*pi*i*r), with r uniform in 0..1.

    |R| = 1 at every bin, so multiplying by it moves no energy at all -- it
    only turns each coefficient in the complex plane.
    """
    return np.exp(2j * np.pi * key_stream(key, purpose, shape))


def symmetric_phase_mask(key, shape):
    """A random phase mask that leaves a real image real.

    A real image has a conjugate-symmetric spectrum, X[-u,-v] = conj(X[u,v]),
    and the inverse transform only comes back real if the mask keeps that. It
    does exactly when its angle is odd:

        phi[-u,-v] = -phi[u,v]

    Drawing r and subtracting its own mirror gives that for free: phi is
    pi * (r - r_mirrored), which flips sign under the mirror by construction,
    and is zero at the bins that are their own mirror (DC, and the Nyquist row
    and column) -- which is where it has to be zero anyway.
    """
    rows, columns = shape
    r = key_stream(key, "frequency-symmetric", shape)
    u = (-np.arange(rows)) % rows
    v = (-np.arange(columns)) % columns
    mirrored = r[u[:, None], v[None, :]]
    return np.exp(1j * np.pi * (r - mirrored))


def masks(key, scheme, shape):
    """The two masks a passphrase stands for: one over the image, one over the spectrum.

    Returned in the order they are applied when encrypting. "phase" has no
    image-side mask -- that is what keeps its ciphertext real -- so it gets a
    mask of ones, and one code path then runs for both ciphers.
    """
    if scheme == "phase":
        return np.ones(shape), symmetric_phase_mask(key, shape)
    return phase_mask(key, "spatial", shape), phase_mask(key, "frequency", shape)
