import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from sequence_jacobian.classes.sparse_jacobians import make_matrix


def winding_number_of_block(block, ss, inputs, outputs, T=600, plot=False):
    return from_H_U(block.jacobian(ss, inputs, outputs, T),
                                        inputs, outputs, plot)

def pack_jacdict_center(J, inputs, outputs):
    """Given a JacobianDict J, extract 'asymptotic' column for each pair
    in inputs and outputs, and pack into Tau*I*O array."""
    T = J[outputs[0], inputs[0]].shape[0]
    center = T//2-1
    ni, no = len(inputs), len(outputs)
    A = np.zeros((2*center+1, no, ni))
    for i, I in enumerate(inputs):
        for o, O in enumerate(outputs):
            J_oi = J[O].get(I)
            if J_oi is not None:
                A[:, o, i] = make_matrix(J_oi, T)[:2*center+1, center]
    return A


def from_H_U(H_U, inputs, outputs, plot=False):
    return winding_number(pack_jacdict_center(H_U, inputs, outputs), plot=plot)


def winding_number(A, Ninterp=8192, plot=False):
    e = sample_values(A, Ninterp)
    if plot:
        plt.plot(e.real, e.imag)
        plt.plot([0], [0], marker='o', markersize=5, color="red")
    return winding_number_of_path(e.real, e.imag)


def sample_values(A, Ninterp=8192):
    """Evaluate Laurent polynomial det A(z) (with equally many positive
    and negative powers) counterclockwise at Ninterp evenly spaced roots of
    unity z, wrapping back around to z=1"""
    # A is (2*Tau-1, n, n) array of Laurent polynomial coefficients
    assert Ninterp % 2 == 0, len(A) % 2 == 1
    Tau = len(A) // 2 + 1

    # n is number of variables, or None if A is scalar
    n = A.shape[1] if A.ndim == 3 else None
    if n is not None:
        assert n == A.shape[2]

    # center A(z) at Ninterp/2
    AA = np.zeros((Ninterp,n,n)) if n is not None else np.zeros(Ninterp)
    AA[Ninterp//2-Tau+1:Ninterp//2+Tau] = A

    # take FFT to evaluate at clockwise roots of unity
    # multiply by alternating 1, -1 to divide by z^(n-2)
    e = np.fft.fft(AA, axis=0)
    alt = np.tile([1, -1], Ninterp//2)
    if n is not None:
        e = np.linalg.det(e*alt[:, np.newaxis, np.newaxis])
    else:
        e = e * alt

    # return wrapped back to z=1, reversed to make counterclocwise
    return np.concatenate((e, [e[0]]))[::-1]


@njit
def winding_number_of_path(x, y):
    """Compute winding number around origin of (x,y) coordinates that make closed path by
    counting number of counterclockwise crossings of ray from (0,0) -> (infty,0) on x axis"""
    # ensure closed path!
    assert x[-1] == x[0] and y[-1] == y[0]

    winding_number = 0

    # we iterate through coordinates (x[i], y[i]), where cur_sign is flag for
    # whether current coordinate is above the x axis
    cur_sign = (y[0] >= 0)
    for i in range(1, len(x)):
        if (y[i] >= 0) != cur_sign:
            # if we're here, this means the x axis has been crossed
            # this generally happens rarely, so efficiency no biggie
            cur_sign = (y[i] >= 0)

            # crossing of x axis implies possible crossing of ray (0,0) -> (infty,0)
            # we will evaluate three possible cases to see if this is indeed the case
            if x[i] > 0 and x[i - 1] > 0:
                # case 1: both (x[i-1],y[i-1]) and (x[i],y[i]) on right half-plane, definite crossing
                # increment winding number if counterclockwise (negative to positive y)
                # decrement winding number if clockwise (positive to negative y)
                winding_number += 2 * cur_sign - 1
            elif not (x[i] <= 0 and x[i - 1] <= 0):
                # here we've ruled out case 2: both (x[i-1],y[i-1]) and (x[i],y[i]) in left 
                # half-plane, where there is definitely no crossing

                # thus we're in ambiguous case 3, where points (x[i-1],y[i-1]) and (x[i],y[i]) in
                # different half-planes: here we must analytically check whether we crossed
                # x-axis to the right or the left of the origin
                # [this step is intended to be rare]
                cross_coord = (x[i - 1] * y[i] - x[i] * y[i - 1]) / (y[i] - y[i - 1])
                if cross_coord > 0:
                    winding_number += 2 * cur_sign - 1
    return winding_number


