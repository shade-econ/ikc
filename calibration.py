import numpy as np
from capital_sticky_prices import production_pricesetting
import json

"""Calibration parameters for models and IKC / quantitative environment
Dicts should be accessed by end user via functions, which return separate
copies to avoid accidentally modifying a shared dict."""

# core parameters consistent across all models and environments
r = 0.05        # annual real interest rate
eis = 1         # elasticity of intertemporal substitution
theta = 0.181   # progressivity of HSV
core = dict(r = r, eis = eis, theta = theta)

# parameters for fiscal shocks
rhoG = 0.76         # persistence of G shock (always used)
rhoB = 0.93         # persistence of B shock (maximum used)

# shared calibration parameters for HA models
calibration_ha = dict(
    theta = theta,
    rho_e = 0.91,           # persistence of idiosyncratic productivity shock
    sd_e = (1-theta)*0.92,  # stdev of post-tax idiosyncratic productivity
)

# specific parameters for HA-one model
calibration_ha_one = dict(
    min_a = 0.,    # min asset on grid
    max_a = 1000,  # max asset on grid
    n_a = 200,     # number of asset grid points
    n_e = 11,      # number of productivity grid points
)

# specific parameters for HA-two model
gamma = 5 
calibration_ha_two = dict(
    zeta = 0.08/(1+r),  # 8% spread
    min_a = 0.,         # min illiquid asset on grid
    max_a = 10000,      # max illiquid asset on grid
    min_b = 0.,         # min liquid asset on grid
    max_b = 10000,      # max liquid asset on grid
    n_a = 50,           # number of illiquid asset grid points
    n_b = 50,           # number of liquid asset grid points
    n_e = 11,           # number of productivity grid points
)

def get_ha_calibrations():
    """Return copies of calibration info for HA-one and HA-two models"""
    return ({**core, **calibration_ha, **calibration_ha_one},
            {**core, **calibration_ha, **calibration_ha_two})

# For consistency, we calibrate the IKC environment to have the same post-tax
# labor income Z and assets A as in the quantitative environment (whenever possible)
# This requires part of the quantitative environment to be calibrated first.
Y_quant = 1.        # output normalized to 1 in quantitative environment
alpha = 0.294       # capital share
delta = 0.08        # depreciation rate
mup = 1.000001      # price markup
B_Y_quant = 0.7     # debt to GDP in quantitative environment
G_Y = 0.2           # government spending to GDP

# total value of assets (in quantitative and most IKC environments)
K_Y = alpha/mup/(r+delta)       # capital-to-GDP
Pi_Y = (1-1/mup)/r              # capitalized profits-to-GDP
A = (B_Y_quant + K_Y + Pi_Y)*Y_quant

# total post-tax labor income (in quantitative and most IKC environments)
Z = ((1-alpha)/mup - G_Y - r*B_Y_quant)*Y_quant

# implied Y in IKC environment (infer from C = Z + r*A and G_Y)
Y_ikc = (Z + r*A)/(1-G_Y)

def get_calibration_ikc():
    """Return copy of calibration info for IKC environment"""
    return dict(
        r = r, eis = eis,
        Y = Y_ikc, G = G_Y*Y_ikc,
        Z = Z, B = A, T = G_Y*Y_ikc + r*A
    )

# == Calibrate rest of quantitative environment ==

# First, price and wage-setting
freq_p = 0.67       # frequency of price adjustment
freq_w = 0.33       # frequency of wage adjustment
Gamma = 5           # real rigidity coefficient

# slope of price Phillips curve (note that 1/(1+r) is used here for discounting)
kappap = 1/(1 + Gamma) * (1 - (1/(1 + r)) * (1 - freq_p)) * freq_p/(1 - freq_p)

# for slope of wage Phillips curve, we need a household beta, which is model-dependent
# to keep kappa_w consistent across models, we use the beta from the HA-two model
# load beta from solved_params.json file
with open('solved_params.json') as f:
    solved_params = json.load(f)
    beta_HA_two = solved_params['HA-two']['beta']

kappaw = 1/(1 + Gamma) * (1 - beta_HA_two * (1 - freq_w)) * freq_w/(1 - freq_w)

# Rest of price-setting and production parameters
N = 1                                             # aggregate labor normalized to 1 in quantitative environment
K = K_Y*Y_quant                                   # aggregate capital
Theta = Y_quant / (K ** alpha * N ** (1-alpha))   # implied Cobb-Douglas productivity

calibration_prod = dict(
    Y = Y_quant, K = K, Theta = Theta, r = r, kappap = kappap, N = N,
    w = (1-alpha)/mup*Y_quant/N,    # implied real wage
    Q = 1,                          # end of period value per unit of capital
    alpha = alpha,                  # capital share
    delta = delta,                  # depreciation rate
    mup = mup,                      # price markup
    epsI = 4,                       # elasticity of investment to Q
    pi = 0,                         # inflation rate
)

# run production_pricesetting.steady_state to make sure these parameters are consistent
ss_prod = production_pricesetting.steady_state(calibration_prod)
assert all(np.isclose(x, 0) for x in (ss_prod['inv'], ss_prod['val'], ss_prod['nkpc'],
                ss_prod['mc'] - 1/mup, ss_prod['N'] - N, ss_prod['J_end_period'] - (K_Y + Pi_Y)*Y_quant))

# finally, merge these with other parameters needed for full quantitative environment
calibration_quant = {**core, **calibration_prod,
                     'kappaw': kappaw,
                     'G': G_Y*Y_quant,
                     'B': B_Y_quant*Y_quant,
                     'Z': Z, 'A': A,
                     'frisch': 1,       # Frisch elasticity of labor supply
                     'muw': 1.001,      # wage markup,
                     'phi_pi': 1.5,     # Taylor rule coefficient
}

def get_calibration_quant():
    return calibration_quant.copy()
