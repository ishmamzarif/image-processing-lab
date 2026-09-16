"""Timing the hand-written version against the library one.

The page shows both times and how many times faster the library was. The
hand-written side is always slower -- it is plain Python loops against C -- and
the ratio is there to make that cost visible, not to be beaten.
"""

import time


def timed(function, *args):
    """Call function(*args) and return (its result, seconds it took)."""
    start = time.time()
    result = function(*args)
    return result, time.time() - start


def timing_fields(elapsed, elapsed_lib):
    """The timing values every result page shows, formatted for the template."""
    return {
        "elapsed": "{:.2f}".format(elapsed),
        "elapsed_lib": "{:.4f}".format(elapsed_lib),
        "speedup": "{:.0f}".format(elapsed / elapsed_lib) if elapsed_lib > 0 else None,
    }
