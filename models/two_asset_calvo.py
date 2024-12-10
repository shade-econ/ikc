import numpy as np
from numba import guvectorize, njit

from sequence_jacobian.interpolate import interpolate_coord, interpolate_y
import sequence_jacobian as sj

@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(np),(np)->(n)')
def interpolate(x, xp, yp, y):
    n_p = len(xp)
    n = len(x)

    for iq in range(n):
        if x[iq] < xp[0]:
            ilow = 0
        elif x[iq] > xp[-2]:
            ilow = n_p-2
        else:
            # start binary search
            # should end with ilow and ihigh exactly 1 apart, bracketing variable
            ihigh = n_p-1
            ilow = 0
            while ihigh - ilow > 1:
                imid = (ihigh + ilow) // 2
                if x[iq] > xp[imid]:
                    ilow = imid
                else:
                    ihigh = imid

        pi = (xp[ilow+1] - x[iq]) / (xp[ilow+1] - xp[ilow])
        y[iq] = pi*yp[ilow] + (1-pi)*yp[ilow+1]


def backward_oneasset_core(Va, a_grid, y, eis, already_cspace=False):
    """Given marginal value of savings Va[s,a], find optimal
    consumption vs. assets today"""
    c_endog = Va**(-eis) if not already_cspace else Va 
    coh = y[:, np.newaxis] + a_grid
    a = interpolate_y(c_endog + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])
    c = coh - a
    return a, c


def enforce_constraints(a, abar, bbar, total):
    """Given a proposed split of 'total' into 'a' and residual 'b', enforce constraint
    that a>=abar and b>bbar (assume this is always possible, but prioritize b)"""
    a = np.maximum(a, abar)
    b = total - a
    b = np.maximum(b, bbar)
    a = total - b
    return a, b


def interpolate_2d_special(x, y, xp, yp, zps, shp=None):
    """Given grids xp and yp producing various outputs zps,
    monotonic x and y we want to evaluate, and an extra leading
    dimension on x and y, figure out the zs. If needed, broadcast to some shape"""
    if shp is None:
        assert x.shape == y.shape
        shp = x.shape
    else:
        x = np.broadcast_to(x, shp)
        y = np.broadcast_to(y, shp)
    
    xi, xpi = interpolate_coord(xp, x)
    yi, ypi = interpolate_coord(yp, y)

    zs = []
    for zp in zps:
        z = expectation_policy_2d_alt(zp, xi.reshape(shp[0], -1), yi.reshape(shp[0], -1),
                                          xpi.reshape(shp[0], -1), ypi.reshape(shp[0], -1))
        zs.append(z.reshape(shp))
    return zs


@njit
def expectation_policy_2d_alt(X, x_i, y_i, x_pi, y_pi):
    nZ, nW = x_i.shape
    Xnew = np.empty(x_i.shape)
    for iz in range(nZ):
        for iw in range(nW):
            ixp = x_i[iz, iw]
            iyp = y_i[iz, iw]
            alpha = x_pi[iz, iw]
            beta = y_pi[iz, iw]

            Xnew[iz, iw] = (alpha * beta * X[iz, ixp, iyp] + alpha * (1-beta) * X[iz, ixp, iyp+1] +
                                (1-alpha) * beta * X[iz, ixp+1, iyp] +
                                (1-alpha) * (1-beta) * X[iz, ixp+1, iyp+1])
    return Xnew

def twoasset_calvo_init(a_grid, b_grid, y, eis):
    Vb = (0.05*a_grid[:, np.newaxis] + 0.2*b_grid + 0.2*y[:, np.newaxis, np.newaxis])**(-eis)
    Va = 0.8*Vb + (0.5 + 0.06*a_grid[:, np.newaxis])**(-eis)
    Vb = np.tile(Vb, (2,1,1,1))
    Va = np.tile(Va, (2,1,1,1))
    return Va, Vb

@sj.het(exogenous=('Pi_calvo', 'Pi'), policy=('a', 'b'), backward=('Va', 'Vb'), backward_init=twoasset_calvo_init)
def twoasset_calvo(Va_p, Vb_p, a_grid, b_grid, y, ra, rb, beta, eis):
    # Calvo is iid so e.g. Va_p[0], Va_p[1] should be identical
    Wa_c = (beta*Va_p[0])**(-eis)
    Wb_c = (beta*Vb_p[0])**(-eis)
    shp = (len(y), len(a_grid), len(b_grid))
    
    # step 1a: find mapping from a to total that makes us indifferent between a and b
    total_from_a = interpolate(np.array([0]), Wb_c - Wa_c, b_grid).reshape(shp[:2]) + a_grid
    
    # step 1b: invert this to get mapping from each current (s, total) to ideal (a,b)
    # (use a_grid for totals too, let's assume this is big enough?)
    a_ideal = interpolate(a_grid, total_from_a, a_grid)
    a_ideal, b_ideal = enforce_constraints(a_ideal, a_grid[0], b_grid[0], a_grid)
    
    # step 1c: get common marginal value if we do adjust
    Vx_c_ideal, = interpolate_2d_special(a_ideal, b_ideal,
                      a_grid, b_grid, [np.minimum(Wa_c, Wb_c)])
    
    
    # step 2a: basic one-asset iteration if you can't adjust
    b, c = backward_oneasset_core(Wb_c.reshape(-1, len(b_grid)), b_grid,
                            np.repeat(y, len(a_grid)), eis, already_cspace=True)
    b, c = b.reshape(shp), c.reshape(shp)
    
    # step 2b: basic one-asset iteration if you will adjust
    # (still as function of incoming *total* assets)
    total_adjust, c_adjust = backward_oneasset_core(Vx_c_ideal, a_grid,
                            y, eis, already_cspace=True)
    a_adjust = interpolate(total_adjust, a_grid, a_ideal)
    b_adjust = interpolate(total_adjust, a_grid, b_ideal)
    
    # step 2c: what are a, b, and c if we'll adjust as functions of incoming asset states?
    totals = (a_grid[:, np.newaxis] + b_grid).ravel()
    a_adjust = interpolate(totals, a_grid, a_adjust).reshape(shp)
    b_adjust = interpolate(totals, a_grid, b_adjust).reshape(shp)
    c_adjust = interpolate(totals, a_grid, c_adjust).reshape(shp)
    
    # step 2d: what are marginal values in each case
    Va_c = interpolate(b, b_grid, Wa_c)
    Vb_c = c
    Vx_c_adjust = c_adjust
          
    # step 4: move everything from post-return to pre-return grid
    a = np.broadcast_to((1+ra)*a_grid[:, np.newaxis], shp)
    (Va_c, Vb_c, c, b, Vx_c_adjust, c_adjust, a_adjust, b_adjust) = interpolate_2d_special(
                    (1+ra)*a_grid[:, np.newaxis], (1+rb)*b_grid, a_grid, b_grid,
                    [Va_c, Vb_c, c, b, Vx_c_adjust, c_adjust, a_adjust, b_adjust], shp=shp)

    # step 5: assemble into full arrays
    Va = (1+ra)*np.stack([Vx_c_adjust, Va_c])**(-1/eis)
    Vb = (1+rb)*np.stack([Vx_c_adjust, Vb_c])**(-1/eis)
    c = np.stack([c_adjust, c])
    b = np.stack([b_adjust, b])
    a = np.stack([a_adjust, a])
    
    return Va, Vb, c, b, a

def calvo(nu):
    Pi_calvo = np.array([[nu, 1-nu],
                         [nu, 1-nu]])
    return Pi_calvo

def make_grids_twoassets(rho_e, sd_e, n_e, min_a, max_a, n_a, min_b, max_b, n_b):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    b_grid = sj.grids.asset_grid(min_b, max_b, n_b)
    return e_grid, Pi, a_grid, b_grid