"""
Example 1: Solving a continuous-time algebraic
Riccati equation (CARE) using the EBA4 method.
"""

import argparse

from _fdm_2d import fdm_2d_matrix

import matplotlib.pyplot as plt
import numpy as np

from eba4carepy import solve_care, integrate_care


def create_problem(n0, s, p, seed, fx, fy, g):
    """
    Create a problem for the CARE solver.
    """
    np.random.seed(seed)
    a = fdm_2d_matrix(n0, fx, fy, g)
    a = a.toarray()
    b = np.random.rand(n0**2, p)
    c = np.random.rand(s, n0**2)
    z0 = np.random.rand(n0**2, s)
    return a, b, c, z0


def main(check=False, **kwargs):
    """
    Main function for the example.
    """
    problem_params = {
        'n0': 10,
        's': 2,
        'p': 2,
        'seed': 34,
        'fx': lambda x, y: 10 * x + y,
        'fy': lambda x, y: np.exp(x**2 * y),
        'g': lambda x, y: 20*y
    }
    A, B, C, Z0 = create_problem(**problem_params)
    t0 = 0.0
    tf = 1.0
    h = 0.01
    mm = 20
    rtol = 1e-12
    atol = 1e-16
    verbose = kwargs.get('verbose', False)
    eba_result = solve_care(
        A, B, C,
        t0, tf, h,
        Z0,
        mm,
        rtol=rtol,
        atol=atol,
        verbose=verbose,
        check=check
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
        A=A, B=B, Q=Q, Y0=Z0 @ Z0.T,
        t0=t0, tf=tf, t=eba_result.t,
        method='RK23',
        rtol=rtol,
        atol=atol
    )
    X = ode_result.y
    Ym = eba_result.Y
    vm = eba_result.V
    Xm = vm @ Ym @ vm.T

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(check=args.check, verbose=args.verbose)
