import argparse
from _fdm_2d import fdm_2d_matrix
import matplotlib.pyplot as plt
import numpy as np
from eba4carepy import solve_care, integrate_care
import time
from ex1 import create_problem


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
        'g': lambda x, y: 20 * y
    }

    # Time checkpoint for problem creation
    start_time = time.time()
    A, B, C, Z0 = create_problem(**problem_params)
    end_time = time.time()
    print(f"Problem creation time: {end_time - start_time:.6f} seconds")

    t0 = 0.0
    tf = 1.0
    h = 0.001
    mm = 20
    rtol = 1e-12
    atol = 1e-16
    verbose = kwargs.get('verbose', False)

    # Time checkpoint for solving CARE
    start_time = time.time()
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
    end_time = time.time()
    print(f"CARE solving time: {end_time - start_time:.6f} seconds")

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

    # Time checkpoint for true CARE solution
    start_time = time.time()
    Q = C.T @ C
    ode_result = integrate_care(
        A=A, B=B, Q=Q, Y0=Z0 @ Z0.T,
        t0=t0, tf=tf, t=eba_result.t,
        method='RK23',
        rtol=rtol,
        atol=atol
    )
    end_time = time.time()
    print(f"True CARE solution time: {end_time - start_time:.6f} seconds")

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
