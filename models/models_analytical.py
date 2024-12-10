import numpy as np
from scipy import linalg
import sequence_jacobian as sj

from base import calibration, jacobian_manipulation as jac


"""Obtain M and A Jacobian matrices for analytical models RA, TA, BU, TABU, ZL given parameters"""

def MA_all(params, r, T):
    """Obtain M and A matrices for all analytical models given parameters"""
    M, A = {}, {}
    M['RA'], A['RA'] = MA_RA(r, T)
    M['TA'], A['TA'] = MA_TA(r, params['TA']['mu'], T)
    M['BU'], A['BU'] = MA_BU(params['BU']['beta'], r, params['BU']['lamb'], T)
    M['TABU'], A['TABU'] = MA_TABU(params['TABU']['beta'], r, params['TABU']['lamb'], params['TABU']['mu'], T)
    M['ZL'], A['ZL'] = MA_ZL(params['ZL']['beta'], r, params['ZL']['lamb'], params['ZL']['mu'], T)
    return M, A


def MA_RA(r, T):
    # use equation (A.60) for asset Jacobian
    beta = 1/(1+r)

    # upper and lower triangles of A
    q = beta**np.arange(T)
    A_U = -(1-beta*q)[:, np.newaxis] * linalg.toeplitz(beta**(np.arange(T)))
    A_L = np.outer(np.ones(T), beta*q)
    A = np.triu(A_U, 1) + np.tril(A_L)

    # standard equation (23) for M^RA
    M = (1-beta)*np.outer(np.ones(T), q)    
    # assert np.allclose(M, jac.M_from_A(A, r), atol=1E-10) # check consistency

    return M, A


def MA_TA(r, mu, T):
    M_RA, A_RA = MA_RA(r, T)
    M = mu*np.eye(T) + (1-mu)*M_RA
    A = (1-mu)*A_RA
    # assert np.allclose(M, jac.M_from_A(A, r), atol=1E-10) # check consistency

    return M, A


def MA_BU(beta, r, lamb, T):
    # implement equation (A.67) for A
    left_col = lamb**np.arange(T)
    right_row = -(1 - lamb/(1+r))*(beta*lamb)**np.arange(T)
    right_row[0] = lamb/(1+r)

    # equivalent: make fake news matrix as outer product
    A = jac.J_from_F(np.outer(left_col, right_row))
    M = jac.M_from_A(A, r)

    # verify that this is the same as clunkier direct implementation of (A.67)
    # assert np.allclose(A, np.tril(linalg.toeplitz(left_col)) @ np.triu(linalg.toeplitz(right_row)))
    
    return M, A


def MA_TABU(beta, r, lamb, mu, T):
    M_BU, A_BU = MA_BU(beta, r, lamb, T)
    M = mu*np.eye(T) + (1-mu)*M_BU
    A = (1-mu)*A_BU
    # assert np.allclose(M, jac.M_from_A(A, r), atol=1E-10) # check consistency

    return M, A


def MA_ZL(beta, r, lamb, mu, T):
    # implement equation (A.93) for A
    left_col = (1-mu)*lamb**np.arange(T)
    right_row = -(1/beta-lamb)/(1+r)*(beta*lamb)**np.arange(T)
    right_row[0] = lamb/(1+r)

    # equivalent: make fake news matrix as outer product
    A = jac.J_from_F(np.outer(left_col, right_row))
    M = jac.M_from_A(A, r)

    # verify that this is the same as clunkier direct implementation of (A.93)
    # assert np.allclose(A, np.tril(linalg.toeplitz(left_col)) @ np.triu(linalg.toeplitz(right_row)))

    return M, A


"""Return SSJ blocks (other than ZL) and corresponding steady states for quantitative / nonlinear analysis"""
# analogous to get_* routines in models_heterogeneous.py

core = dict(rpost=calibration.r, Z=calibration.Z, A=calibration.A,
            C=calibration.Z + calibration.r*calibration.A, eis=calibration.eis)

def get_all_quant(params):
    hh_analytical, ss_analytical = {}, {}
    hh_analytical['RA'], ss_analytical['RA'] = get_RA(params['RA'])
    hh_analytical['TA'], ss_analytical['TA'] = get_TA(params['TA'])
    hh_analytical['BU'], ss_analytical['BU'] = get_BU(params['BU'])
    hh_analytical['TABU'], ss_analytical['TABU'] = get_TABU(params['TABU'])
    return hh_analytical, ss_analytical

def get_RA(param):
    ss = hh_ra.steady_state({**core, 'beta': 1/(1+calibration.r)}, dissolve=['hh_ra'])
    assert np.isclose(ss['euler'], 0) and np.isclose(ss['budget_constraint'], 0)
    return hh_ra, ss

def get_TA(param):
    calib = {**core, 'beta': 1/(1+core['rpost']), 'mu': param['mu'],
             'C_RA': core['Z'] + core['rpost']*core['A']/(1-param['mu'])}
    ss = hh_ta.steady_state(calib, dissolve=['hh_ta'])
    assert np.isclose(ss['euler'], 0) and np.isclose(ss['budget_constraint'], 0)
    return hh_ta, ss

def get_BU(param):
    calib = {**core, 'beta': param['beta'], 'lamb': param['lamb'], 'Ass': core['A']}
    calib.update(get_bu_param(calib['beta'], calib['rpost'], calib['lamb'], calib['eis'], calib['C']))
    ss = hh_bu.steady_state(calib)
    return hh_bu, ss

def get_TABU(param):
    calib = {**core, 'beta': param['beta'], 'lamb': param['lamb'], 'mu': param['mu'], 'Ass': core['A']/(1-param['mu'])}
    calib.update(get_bu_param(calib['beta'], calib['rpost'], calib['lamb'], calib['eis'],
                              calib['Z'] + calib['rpost']*calib['Ass']))
    ss = hh_tabu.steady_state(calib)
    return hh_tabu, ss


def get_bu_param(beta, r, lamb, eis, C):
    """Obtain the chip and chipp parameters for BU model"""
    # use equations below (A.77) in paper
    up = C**(-1/eis)
    upp = -1/eis*C**(-1/eis-1)

    chip = up*(1-beta*(1+r))
    chipp = upp*(1 + r - lamb)*(1 - beta*(1+r)*lamb) / lamb
    return {'chip': chip, 'chipp': chipp}


"""SSJ blocks representing analytical models (other than ZL)"""
# note: these report "UCE", which here is just average marginal utility of consumption
# (weighted by labor earnings for het-agent models, but no labor earning heterogeneity here)

@sj.solved(unknowns={'C': 1, 'A': 1},
           targets=["euler", "budget_constraint"])
def hh_ra(C, A, Z, eis, beta, rpost):
    euler = (beta * (1 + rpost(+1)))**(-eis) * C(+1) - C
    budget_constraint = (1 + rpost) * A(-1) + Z - C - A
    UCE = C**(-1/eis)
    return euler, budget_constraint, UCE


@sj.solved(unknowns={'C_RA': 1, 'A': 1},
           targets=["euler", "budget_constraint"])
def hh_ta(C_RA, A, Z, eis, beta, rpost, mu):
    euler = (beta * (1 + rpost(+1))) ** (-eis) * C_RA(+1) - C_RA
    C_H2M = Z
    C = (1 - mu) * C_RA + mu * C_H2M
    budget_constraint = (1 + rpost) * A(-1) + Z - C - A
    UCE = (1 - mu) * C_RA**(-1/eis) + mu * C_H2M**(-1/eis)
    return euler, budget_constraint, C, UCE


@sj.solved(unknowns={'C': 0.8, 'A': 5},
           targets=['euler', 'budget_constraint']) 
def hh_bu(C, A, Z, rpost, beta, eis, chip, chipp, Ass):
    euler = beta * (1 + rpost(1)) * C(1) ** (-1 / eis) + chip + chipp*(A-Ass)  - C ** (-1/eis)
    budget_constraint = (1 + rpost) * A(-1) + Z - C - A
    UCE = C**(-1/eis)
    return euler, budget_constraint, UCE


@sj.solved(unknowns={'C_BU': 0.8, 'A_BU': 5},
           targets=['euler_BU', 'budget_constraint_BU']) 
def hh_tabu(C_BU, A_BU, Z, rpost, beta, eis, chip, chipp, Ass, mu):
    euler_BU = beta * (1 + rpost(1)) * C_BU(1) ** (-1 / eis) + chip  + chipp*(A_BU-Ass)  - C_BU ** (-1/eis)
    budget_constraint_BU = (1 + rpost) * A_BU(-1) + Z - C_BU - A_BU
    A = (1 - mu) * A_BU 
    C_H2M = Z 
    C = (1 - mu) * C_BU + mu * C_H2M
    UCE =  (1 - mu) * C_BU**(-1/eis) + mu * C_H2M**(-1/eis)
    return euler_BU, budget_constraint_BU, A, C, UCE


