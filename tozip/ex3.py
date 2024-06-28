"""
Example 3: Riccati EBA with BDF integration
"""

import numpy as np
import matplotlib.pyplot as plt

from eba4carepy import solve_care, integrate_care


def create_problem(n, s, alpha, dt, seed):
    """
    Create a problem for the CARE solver.
    """
    np.random.seed(seed)
    f = np.random.rand(n, s)
    c = np.random.rand(s, n)
    m = (1/(6*n)) * (4*np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1))
    km = -(alpha * n) * (2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    a = -np.linalg.solve((m-dt*km), m)
    b = dt * np.linalg.solve((m-dt*km), f)
    return a, b, c, np.zeros((n, s))


def main():
    """
    Main function for the example.
    """
    problem_params = {
        'n': 49,
        's': 2,
        'alpha': 1.0,
        'dt': 0.01,
        'seed': 34
    }
    A, B, C, Z0 = create_problem(**problem_params)
    t0 = 0.0
    tf = 2.0
    h = 0.01
    mm = 10
    rtol = 1e-8
    atol = 1e-10
    verbose = True
    eba_result = solve_care(
        A, B, C,
        t0, tf, h,
        Z0,
        mm,
        rtol=rtol,
        atol=atol,
        verbose=verbose,
        check=True
    )
    rrn = eba_result.rrn
    plt.plot(
        range(len(rrn)),
        rrn,
        '-o',
        label='Relative error'
    )
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.yscale('log')
    plt.legend()

    # Solve the true CARE problem.
    Q = C.T @ C
    ode_result = integrate_care(
        A=A, B=B, Q=Q,
        t0=t0, tf=tf, t=eba_result.t,
        Y0=Z0 @ Z0.T,
        rtol=rtol,
        atol=atol
    )
    X = ode_result.y
    Ym = eba_result.Y
    Vm = eba_result.V
    Xm = Vm @ Ym @ Vm.T

    err = np.linalg.norm(Xm - X, axis=(1, 2))
    plt.figure()
    plt.plot(
        eba_result.t,
        err,
        label='Error'
    )
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
