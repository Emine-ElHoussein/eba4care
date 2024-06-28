"""
Implementation of the Extended Block Arnoldi method for the
solving the Continuous Algebraic Riccati Equation (CARE).

The module provides the following functions:

    * care: Compute the Continuous Algebraic Riccati
        Equation (CARE).
    * integrate_care: Solve the Continuous Algebraic
        Riccati Equation (CARE) using the
        `scipy.integrate.solve_ivp` function.
    * solve_care: Solve the Continuous Algebraic
        Riccati Equation (CARE) using the Extended
        Block Arnoldi method. The function returns
        an instance of the `CareResult`.
"""

from dataclasses import dataclass
from scipy import integrate
import numpy as np


def _frobenius_norm_squared(M):
    return np.sum(M**2)


def care(A, B, Q, X):
    return A.T @ X + X @ A - X @ B @ B.T @ X + Q


def integrate_care(
    A, B, Q,
    Y0,
    t0, tf, t,
    method='BDF',
    **options
):
    def _jac(_, y):
        Y = y.reshape(A.shape)
        dYdt = care(A=A, B=B, Q=Q, X=Y)
        dydt = dYdt.flatten()
        return dydt

    y0 = Y0.flatten()
    res = integrate.solve_ivp(
        _jac,
        [t0, tf],
        y0,
        t_eval=t,
        method=method,
        **options
    )
    res.y = res.y.T.reshape((-1, *Y0.shape))
    return res


@dataclass
class CareResult:
    t: np.ndarray
    Y: np.ndarray
    V: np.ndarray
    m: int
    rn: list[float]
    rrn: list[float]


def show(msg: str, logger):
    if logger is not None:
        logger(msg)


def solve_care(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    t0: float,
    tf: float,
    h: float,
    Z0: np.ndarray,
    mm: int,
    **options
):
    verbose = options.pop('verbose', False)
    logger = print if verbose else None
    check = options.pop('check', False)
    rtol = options.get('rtol', 1e-3)
    method = options.get('method', 'BDF')

    # Initialize the algorithm
    norm_sq_c = _frobenius_norm_squared(C)
    s, n = C.shape
    _, p = B.shape
    _, r = Z0.shape
    nT = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, nT)

    # Transpose the matrices A and C.
    tA, tC = A.T, C.T

    # Initialize the Krylov method
    V = np.zeros((n, 2*(mm+1)*s))
    H = np.zeros((2*(mm+1)*s, 2*s))
    U = np.zeros((n, 2*s))
    rho = np.zeros((s, s))
    iL = np.zeros((s, s))
    eL = np.zeros((2*s, 2*s))

    # Tm = V{m+1}.T @ tA @ V{m}
    T = np.zeros((2*(mm+1)*s, 2*mm*s))
    Bmm = np.empty((2*mm*s, p))
    Z0mm = np.empty((2*mm*s, r))

    # First step:
    # Compute the QR factorization of the matrix U.
    U[:, :s] = tC
    U[:, s:] = np.linalg.solve(tA, tC)
    V[:, :2*s], L = np.linalg.qr(U)
    iL = L[:s, :s]
    eL = np.linalg.inv(L)
    Qm = iL @ iL.T

    rns = []
    rrns = []

    for m in range(1, mm+1):
        if check:
            msg = f"Checking Vm.T @ Vm = I at m={m}..."
            Vm = V[:, :2*m*s]
            if not np.allclose(Vm.T @ Vm, np.eye(2*m*s)):
                msg += "Failed."
            else:
                msg += "Passed."
            show(msg, logger)

        # Next step:
        idx = slice(2*(m-1)*s, 2*m*s)
        V1, V2 = np.hsplit(V[:, idx], 2)
        U[:, :s] = tA @ V1
        U[:, s:] = np.linalg.solve(tA, V2)

        for i in range(m):
            idx = slice(2*i*s, 2*(i+1)*s)
            H[idx, :] = V[:, idx].T @ U
            U -= V[:, idx] @ H[idx, :]

        # Compute the QR factorization of the matrix U.
        idx = slice(2*m*s, 2*(m+1)*s)
        V[:, idx], H[idx, :] = np.linalg.qr(U)

        # Update the matrix Tm:
        end = 2*(m+1)*s
        odds = slice(2*(m-1)*s, (2*m-1)*s)
        T[:end, odds] = H[:end, :s]

        evens = slice((2*m-1)*s, 2*m*s)
        if m == 1:
            T[:end, evens] = H[:end, :s] @ iL @ eL[:s, s:]
            T[:s, evens] += iL @ eL[s:, s:]
        else:
            T[:end, evens] += T[:end, odds] @ rho

        # Check if Tm = Vm^* @ A^T @ Vm
        if check:
            msg = f"Checking if Tm = Vm.T @ A @ Vm at m={m}..."
            passed = True
            for j in range(m):
                jy = slice(2*j*s, 2*(j+1)*s)
                for i in range(j+1):
                    ix = slice(2*i*s, 2*(i+1)*s)
                    eT = V[:, ix].T @ tA @ V[:, jy]
                    tT = T[ix, jy]
                    if not np.allclose(eT, tT):
                        passed = False
                        msg += "\n" + "Failed at " + f"i={i}, j={j}."
                        print(eT)
                        print(tT)
            if passed:
                msg += "Passed."
            show(msg, logger)

        if m < mm:
            idx = slice(2*m*s, 2*(m+1)*s)
            odds = slice(2*m*s, (2*m+1)*s)
            eL = np.linalg.inv(H[idx, :])
            iL = H[odds, :s]
            rho = iL @ eL[:s, s:]

            rows = slice((2*m-1)*s, 2*m*s)
            cols = slice((2*m+1)*s, 2*(m+1)*s)
            T[rows, cols] += eL[s:, s:]

            end = 2*(m+1)*s
            T[:end, cols] -= T[:end, :2*m*s] @ (H[:2*m*s, s:] @ eL[s:, s:])

        idx = slice(2*(m-1)*s, 2*m*s)
        Bmm[idx, :] = V[:, idx].T @ B
        Z0mm[idx, :] = V[:, idx].T @ Z0

        if m == 1:
            Qm = np.pad(Qm, ((0, s), (0, s)))
        else:
            Qm = np.pad(Qm, ((0, 2*s), (0, 2*s)))
        Am = T[:2*m*s, :2*m*s].T
        Bm = Bmm[:2*m*s, :]
        Z0m = Z0mm[:2*m*s, :]

        # Check if Qm = Cm.T @ Cm
        if check:
            msg = f"Checking if Qm = Cm.T @ Cm at m={m}..."
            if m == 1:
                Cm = C @ V[:, idx]
            else:
                Cm = np.hstack([Cm, C @ V[:, idx]])
            if not np.allclose(Qm, Cm.T @ Cm):
                msg += "Failed."
            else:
                msg += "Passed."
            show(msg, logger)

        # Integrate the projected CARE:
        ode_res = integrate_care(
            A=Am, B=Bm, Q=Qm,
            Y0=Z0m @ Z0m.T,
            t0=t0, tf=tf, t=t,
            method=method,
            **options
        )
        msg = f"Integrating the CARE at m={m}..."
        show(msg, logger)
        msg = f"Integration status: {ode_res.status}"
        show(msg, logger)
        # msg = f"Integration message: {ode_res.message}"
        # show(msg, logger)

        # Compute the norm of the residual.
        Ym = ode_res.y[-1]
        if check:
            msg = f"Checking the residual at m={m}..."
            Q = C.T @ C
            Vm = V[:, :2*m*s]
            dYmdt = care(A=Am, B=Bm, Q=Qm, X=Ym)
            dXmdt = Vm @ dYmdt @ Vm.T
            Xm = Vm @ Ym @ Vm.T
            Rm = dXmdt - care(A=A, B=B, Q=Q, X=Xm)
            if not np.allclose(Vm.T @ Rm @ Vm, 0):
                msg += "Failed."
            else:
                msg += "Passed."
            show(msg, logger)

        rows = slice(2*m*s, 2*(m+1)*s)
        cols = slice(2*(m-1)*s, 2*m*s)
        Rm = 2 * T[rows, cols] @ Ym[-2*s:, :]
        rn = _frobenius_norm_squared(Rm)
        rrn = rn / norm_sq_c
        rns += [rn]
        rrns += [rrn]
        msg = f"m={m}: rn={rn}, rrn={rrn}"
        show(msg, logger)

        if rrn < rtol:
            msg = "Convergence reached.\n"
            msg += f"Relative residual norm: {rrn}"
            show(msg, logger)
            break

    else:
        msg = "Maximum number of iterations reached."
        show(msg, logger)

    return CareResult(
        t=t,
        Y=ode_res.y,
        V=V[:, :2*m*s],
        m=m,
        rn=rns,
        rrn=rrns
    )
