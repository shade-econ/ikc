"""Define HA-one and HA-two model blocks."""

import numpy as np
import sequence_jacobian as sj
from .two_account_calvo import twoasset_calvo, calvo, make_grids_twoassets
from base import calibration

"""HA-one model"""

make_grids = sj.hetblocks.hh_sim.make_grids

def income(Z, e_grid):
    y = Z * e_grid
    return y

ha_one = sj.hetblocks.hh_sim.hh.add_hetinputs([make_grids, income])


"""HA-two model"""

hh_two = twoasset_calvo.add_hetinputs([make_grids_twoassets, calvo, income])
hh_two = hh_two.remap({'C': 'Cspend', 'A':'Ailliq', 'B':'Aliq'})

@sj.simple
def cons_assets_total(Cspend, Ailliq, Aliq, zeta,r):
    C = Cspend + zeta*(1+r)*Aliq(-1)
    A = Ailliq + Aliq
    return C, A

@sj.simple
def rates_twoasset(r, zeta):
    rb = r - zeta*(1+r)
    ra = r
    return ra, rb

ha_two = sj.create_model([hh_two,  cons_assets_total, rates_twoasset], name='Two Asset Model')


"""Convenience routines to calculate HA-hi-liq, HA-one, HA-two given parameters and calibration"""

def get_all(params):
    calib_ha_one, calib_ha_two = calibration.get_ha_calibrations()

    hh_het, ss_het = {}, {}
    hh_het['HA-hi-liq'], ss_het['HA-hi-liq'] = ha_one, ha_one.steady_state({**calib_ha_one, **params['HA-hi-liq']})
    hh_het['HA-one'], ss_het['HA-one'] = ha_one, ha_one.steady_state({**calib_ha_one, **params['HA-one']})
    hh_het['HA-two'], ss_het['HA-two'] = ha_two, ha_two.steady_state({**calib_ha_two, **params['HA-two']})
    return hh_het, ss_het


"""HA-one model with option for lump-sum transfer (calibrated to zero in steady state)"""
def income_lumpsum(Z, e_grid, Tr):
    y = Z * e_grid + Tr
    return y

ha_one_lumpsum = sj.hetblocks.hh_sim.hh.add_hetinputs([make_grids, income_lumpsum])

def get_ha_one_lumpsum(param):
    calib = calibration.get_ha_calibrations()[0]
    calib = {**calib, **param, 'Tr': 0}
    ss = ha_one_lumpsum.steady_state(calib)
    return ha_one_lumpsum, ss


"""HA-one model with flexible borrowing constraint (calibrated to zero steady in state)"""
# unfortunately, need to rewrite the backward iteration code from SSJ to add this flexibility
@sj.het(exogenous='Pi',  policy='a',  backward='Va', backward_init=sj.hetblocks.hh_sim.hh_init)
def hh_con(Va_p, a_grid, y, r, beta, eis, a_con):
    # identical to sj.hetblocks.hh_sim.hh except replace a_grid[0] with a_con
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_con)
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c

ha_con = hh_con.add_hetinputs([make_grids, income])

def get_ha_one_con(param):
    calib = calibration.get_ha_calibrations()[0]
    calib = {**calib, **param, 'a_con': 0}
    ss = ha_con.steady_state(calib)
    return ha_con, ss


"""Quantitative case, with earnings-weighted average marginal utility, UCE, as extra hetoutput"""

def compute_uce_one(c,e_grid,eis):
    uce = c ** (-1 / eis) * e_grid[:, np.newaxis]
    return uce

def compute_uce_two(c,e_grid,eis):
    uce = c ** (-1 / eis) * e_grid[np.newaxis, :, np.newaxis, np.newaxis]
    return uce

ha_one_uce = ha_one.add_hetoutputs([compute_uce_one])
hh_two_uce = twoasset_calvo.add_hetinputs([make_grids_twoassets, calvo, income]).add_hetoutputs([compute_uce_two])
hh_two_uce = hh_two_uce.remap({'C': 'Cspend', 'A':'Ailliq', 'B':'Aliq'})
ha_two_uce = sj.create_model([hh_two_uce,  cons_assets_total, rates_twoasset], name='Two Asset Model')

def get_all_quant(params):
    # two differences from before: we have the "UCE" output, and we use "rpost" rather than "r" as input
    calib_ha_one, calib_ha_two = calibration.get_ha_calibrations()
    hh_het, ss_het = {}, {}
    hh_het['HA-hi-liq'], ss_het['HA-hi-liq'] = ha_one_uce.remap({'r': 'rpost'}), ha_one_uce.steady_state({**calib_ha_one, **params['HA-hi-liq']})
    hh_het['HA-two'], ss_het['HA-two'] = ha_two_uce.remap({'r': 'rpost'}), ha_two_uce.steady_state({**calib_ha_two, **params['HA-two']})
    return hh_het, ss_het


"""Quantitative case, alternative where liquid account holds only bonds, equity only in illiquid"""
@sj.simple
def rates_twoasset_alt(r, rpost, zeta, liquidshare):
    # replacement for rates_twoasset
    # note that now we're not remapping, so r is the bond return
    rb = r(-1) - zeta*(1+r(-1))                      # liquid account only holds bonds
    ra = (rpost - r(-1)*liquidshare)/(1-liquidshare) # back out illiquid return from total return
    return ra, rb

@sj.simple
def cons_assets_total_alt(Cspend, Ailliq, Aliq, zeta, r):
    # need to change this block too, since intermediation cost proportional to beginning-of-period value
    # which is now just last period's bond rate
    C = Cspend + zeta*(1+r(-1))*Aliq(-1)
    A = Ailliq + Aliq
    return C, A

ha_two_alt = sj.create_model([hh_two_uce,  cons_assets_total_alt, rates_twoasset_alt], name='Two Asset Model')
