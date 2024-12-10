import sequence_jacobian as sj  

## 
# This implements the New Keynesian production side with sticky prices and capital:
# 1. Rotemberg sticky prices 
# 2. Quadratic capital adj. costs
# 3. Cobb-Douglas production
#
#
# Inputs: 
# - Y: amount of output to be produced
# - w: real wage 
# - r: ex-ante real interest rate
#
# Outputs
# - N, mc: labor to be employed, real marginal cost
# - K, I, Q, k_adjust: capital to be used, investment, tobin's Q, capital adjustment costs
# - pi, p_adjust: inflation, price adjustment costs 
# - div, J, J_end_period: dividends, firm value at the beginning (with dividend) and end of the period

# Required inputs with example dict and example call: 
# calibration = {'Y':1,         # Level of output
#                'N':1,         # Level of labor
#                'r':0.05,      # Annual real rate
#                'alpha': 0.3,  # Capital share
#                'delta': 0.08, # Depreciation rate
#                'mup':1.2,     # Price markup (to remove, make this 1.0000001)
#                'epsI':4,      # Elasticity of investment to Q
#                'kappap':0.1,  # Slope of the price phillips curve vs real marginal cost
#               }
# calibration = calibrate_production(calibration)
# ss_prod = production_pricesetting.steady_state(calibration)

# First, a production block returning N, mc, K, Q consistent with Y,w,r
@sj.simple
def labor(Y, w, K, Theta, alpha):
    N = (Y / Theta * 1 / K(-1) ** alpha) ** (1 / (1 - alpha)) # Production function 
    mc = 1/(1-alpha) * w * N / Y                              # Labor FOC
    return N, mc


@sj.simple
def investment(Y, r, mc, K, Q, Theta, delta, epsI, alpha):
    inv = 1/ (delta * epsI) * (K / K(-1) - 1) + 1 - Q        # Investment FOC
    val = mc(+1) * alpha * Y(+1) / K -\
         (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI)) +\
         K(+1) / K * Q(+1) - (1 + r) * Q                     # Dynamic equation for Q
    return inv, val

production = sj.combine([labor, investment])
production_solved = production.solved(unknowns={'Q': 1., 'K': 3.},
                                      targets=['inv', 'val'], solver='broyden_custom')

# Second, a price-setting block returning pi consistent with mc and Y
@sj.solved(unknowns={'pi': (-0.9, 2)}, targets=['nkpc'], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mup*mc - 1) + 1/(1 + r(+1)) * Y(+1) / Y * pi(+1)*(1 + pi(+1)) -\
            pi*(1 + pi) 
    return nkpc

# Third, a block that computes investment and adjustment costs, as well as firm dividends as residuals from the above
@sj.simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
    p_adjust = 1 / ( 2 * (mup - 1) * kappap) * pi ** 2 * Y
    k_adjust = (K / K(-1) - 1) ** 2 / (2 * delta * epsI) * K(-1)
    I = K - (1 - delta) * K(-1) 
    div = Y - w * N - I - p_adjust - k_adjust
    return p_adjust, k_adjust, I, div

# Finally, a block that computes firm value at the beginning and the end of the period
@sj.solved(unknowns={'J': (2, 6)}, targets=['valJ'], solver="brentq")
def val_firm_solved(div, J, r):
    valJ = div + 1/(1 + r)*J(+1) - J 
    J_end_period = 1/(1 + r)*J(+1)
    return valJ, J_end_period

production_pricesetting = sj.combine([production_solved, pricing_solved, dividend, val_firm_solved])
