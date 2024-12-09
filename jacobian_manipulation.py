import numpy as np
from scipy import linalg

"""Part 0: general Jacobian manipulation"""

def F_from_J(J):
    """Fake news matrix from Jacobian"""
    F = J.copy()
    F[1:,1:]-=J[:-1,:-1]
    return F


def J_from_F(F):
    """Jacobian from fake news matrix"""
    J = F.copy()
    for t in range(1, J.shape[1]):
        J[1:, t] += J[:-1, t - 1]
    return J


def M_from_A(A,r):
    """M matrix (consumption Jacobian) from asset Jacobian A"""
    M = np.eye(len(A)) - A
    M[1:] += (1+r)*A[:-1]
    return M

def Kmat(r,T):
    """ Build TxT K matrix with interest r """ 
    q = (1+r)**(-np.arange(T))
    K = np.triu(linalg.toeplitz(-q), 1) 
    return K

def A_from_M(M,r):
    """ Get A from M via K*(I-M) (Warning: not exact for large T) """ 
    T = len(M)
    K = Kmat(r,T)
    return K @ (np.eye(T) - M)


"""Part 1: cognitive discounting a la Gabaix"""

def cognitive_discounting_F(F, delta):
    """Modify fake news matrix F to reflect cognitive discounting delta"""
    # simply shrink column s by delta^s
    return F * (delta ** np.arange(len(F)))


def cognitive_discounting(J, delta):
    """Modify Jacobian J to reflect cognitive discounting delta"""
    return J_from_F(cognitive_discounting_F(F_from_J(J), delta))


"""Part 2: M matrix 'truncation'"""

def truncate_M(M, Tright, Tleft, r):
    """Truncate to 0 if more than Tright below main diagonal or Tleft above"""
    T = len(M)
    Mtrunc = M.copy()
    
    # Truncate
    for s in range(0,T):
        if s > Tleft:
            Mtrunc[:s-Tleft,s] = 0
        if s < T-1-Tright:
            Mtrunc[s+Tright+1:,s] = 0
    
    # Rescale so that column s has PV (1+r)^(-s), i.e. that q'M = q'
    q = (1+r)**(-np.arange(T))
    Mtrunc /= ((q @ Mtrunc) / q)

    # Enforce quasi-Toeplitz to avoid non-invertible I-M (artifact of truncation)
    Mtrunc = enforce_quasi_toeplitz(Mtrunc)

    # get A from M, again enforce quasi-Toeplitz
    Atrunc = enforce_quasi_toeplitz(A_from_M(Mtrunc,r))
    return Mtrunc, Atrunc

def enforce_quasi_toeplitz(J):
    """Ensure quasi-Toeplitz structure by picking middle column
    (assumed sufficiently long-run that correction matrix is zero afterward)
    and repeating it down diagonals."""

    T = len(J)
    lrcol = np.floor(T/2).astype(int) # Pick column in middle
    c = J[:, lrcol]                   # Define long-run column c to be carried over
    J_QT = J.copy()
    J_QT[:,lrcol:] = linalg.toeplitz(c,np.zeros(T-lrcol)) # Enforce exact Toeplitz from c onward
    return J_QT