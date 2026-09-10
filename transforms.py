"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    if(n<=1): return 1
    i = 1
    while(i<n):
        i = i<<1

    return i  

    # raise NotImplementedError("Implement next_power_of_two")


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        N = np.size(x)
        W_N = np.exp(-(2*1j*np.pi*np.arange(N))/N)
        X = np.zeros(N, dtype=np.complex128)
        for k in range(0, N):
            X[k] = 0
            for n in range(0, N):
                X[k] += (x[n]*(W_N[(k*n)%N]))

        return X
        # raise NotImplementedError("Implement DFTAnalyzer.transform")

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        N = np.size(spectrum)
        W_N = np.exp((2*1j*np.pi*np.arange(N))/N)
        x = np.zeros(N, np.complex128)
        for n in range(0, N):
            
            for k in range(0, N):
                x[n] += (spectrum[k]*(W_N[(k*n)%N]))
            x[n] /= N

        return x
        # raise NotImplementedError("Implement DFTAnalyzer.inverse")


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"
       

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method
        N = np.size(x)
        if(next_power_of_two(N)!=N):
            raise ValueError("For FFT, N must be a power of two")
        X = np.array(x, np.complex128)
        N = X.size
        bits = int(np.log2(N))
        for i in range(N):
            j = int(f"{i:0{bits}b}"[::-1], 2)
            if i<j: X[i], X[j] = X[j], X[i]

        for s in range(1, int(np.log2(N))+1):
            M = 2**s
            W_M = np.exp(-1*(2*1j*np.pi)/M)
            for l in range(0, N-M+1, M):
                W = 1
                for k in range(0, M//2):
                    g = X[l+k]
                    h = W*X[l+k+M//2]
                    X[l+k] = g+h
                    X[l+k+M//2] = g-h
                    W *= W_M

        return X

        # raise NotImplementedError("Implement FFTTransformer.transform")

    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        # TODO: implement this method
        N = np.size(spectrum)
        if(next_power_of_two(N)!=N):
            raise ValueError("For FFT, N must be a power of two")
        X = np.array(spectrum, np.complex128)
        return np.conjugate(self.transform(np.conjugate(X))) / N

        # raise NotImplementedError("Implement FFTTransformer.inverse")

        


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        X = np.array(x, dtype=np.complex128)
        N = X.size
        if next_power_of_two(N) == N: return super().transform(X) # radix 2 fft form ei ase
        n = np.arange(N)
        W = np.exp(-1j*np.pi*((n*n)%(2*N))/N)
        M = next_power_of_two(2*N -1)
        a = np.zeros(M, dtype=np.complex128)
        a[:N] = X*W 
        b = np.zeros(M, dtype=np.complex128)
        b[:N] = np.conj(W)
        b[M-N+1:] = np.conj(W)[:0:-1]

        chirpConv = super().inverse(super().transform(a) * super().transform(b)) #conv mane freq domain e multiply
        return W*chirpConv[:N]




        # raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.transform")

    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        X = np.array(spectrum, dtype=np.complex128)
        return np.conj(self.transform(np.conj(X))) / X.size
        # raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.inverse")


class NTTTransformer(FFTTransformer):

    name  = "ntt"
    NTT_PRIME = 998244353
    generator = 3
    base_digits =2
    max_length = 1<<23

    def ntt(self, x, invert):
        p = self.NTT_PRIME
        N = np.size(x)
        if next_power_of_two(N) != N:
            raise ValueError("N must be a power of two for NTT")
        if N>self.max_length:
            raise ValueError("N is larger than 2^23")

        X = np.array(x, dtype=np.int64) % p
        if N == 1:
            return X
        bits = int(np.log2(N))
        for i in range(N):
            j = int(f"{i:0{bits}b}"[::-1], 2)
            if i<j: X[i], X[j] = X[j], X[i]

        for s in range(1, int(np.log2(N))+1):
            M = 2**s
            W_M = pow(self.generator, (p-1)//M, p)
            if invert:
                W_M = pow(W_M, p-2, p)
            W = np.ones(M//2, dtype=np.int64)
            for k in range(1, M//2):
                W[k] = W[k-1]*W_M % p

            blks = X.reshape(-1, M)
            g = blks[:,  :M//2].copy()
            h = blks[:, M//2:]*W%p
            blks[:, :M//2] = (g +h) % p
            blks[:, M//2:] = (g-h)% p

        return X

    def transform(self, x):
        return self.ntt(x, invert=False)

    def inverse(self, spectrum):
        p = self.NTT_PRIME
        X = self.ntt(spectrum, invert=True)
        return X*pow(int(X.size), p-2, p) % p

    def check_base(self, limbs, base_digits):
        bound = limbs * (10 ** base_digits - 1) ** 2
        if bound >= self.NTT_PRIME:
            raise ValueError("limbs in the given base exceeds p")

