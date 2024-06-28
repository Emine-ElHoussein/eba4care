"""
This module provides a function to generate a
finite difference matrix for a 2D problem.
"""
from collections.abc import Callable
from scipy.sparse import csc_matrix

import numpy as np


def fdm_2d_matrix(
    n0: int,
    fx: Callable[[float, float], float],
    fy: Callable[[float, float], float],
    g: Callable[[float, float], float]
):
    """
    Generate a finite difference matrix for the problem:

        -u_xx - u_yy - fx*du/dx - fy*du/dy + g*u = 0  on (0, 1) x (0, 1).
        u = 0 on boundary (0, 1) x (0, 1).
    """
    n = n0 * n0
    h = 1.0 / (n0 + 1)
    t1 = 4.0 / (h**2)
    t2 = -1.0 / (h**2)
    t3 = 1.0 / (2.0 * h)

    length = 5 * n - 4 * n0
    rows = np.zeros(
        length,
        dtype=np.int32
    )
    cols = np.zeros(
        length,
        dtype=np.int32
    )
    data = np.zeros(
        length,
        dtype=np.float64
    )
    p = 0
    row = 0

    for iy in range(1, n0+1):
        y = iy * h
        for ix in range(1, n0+1):
            x = ix * h
            fxv = fx(x, y)
            fyv = fy(x, y)
            gv = g(x, y)

            # A(i, i - n0):
            if iy > 1:
                rows[p] = row
                cols[p] = row - n0
                data[p] = t2 - t3 * fyv
                p += 1

            # A(i, i - 1):
            if ix > 1:
                rows[p] = row
                cols[p] = row - 1
                data[p] = t2 - t3 * fxv
                p += 1

            # A(i, i):
            rows[p] = row
            cols[p] = row
            data[p] = t1 + gv
            p += 1

            # A(i, i + 1):
            if ix < n0:
                rows[p] = row
                cols[p] = row + 1
                data[p] = t2 + t3 * fxv
                p += 1

            # A(i, i + n0):
            if iy < n0:
                rows[p] = row
                cols[p] = row + n0
                data[p] = t2 + t3 * fyv
                p += 1
            row += 1
    a = csc_matrix(
        (-data, (rows, cols)),
        shape=(n, n),
        dtype=np.float64
    )
    return a


def main():
    """
    Main function for the example.
    """
    def fx(x, y):
        return x + 10 * (y**2)

    def fy(x, y):
        return np.sqrt(2*x**2 + y**2)

    def g(x, y):
        return x * y

    a = fdm_2d_matrix(2, fx, fy, g)
    print(a.toarray())


if __name__ == '__main__':
    main()
