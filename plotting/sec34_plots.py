import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from models.two_asset_calvo import interpolate # To interpolate cdf
from scipy.integrate import trapz       # To calculate Gini

plt.rc('font', size=15)

def set_texfig(fontsize=12.):
    """Defaults when using texfig."""
    plt.rc('text', usetex=True)
    plt.rc('font', family = 'serif')
    plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
    plt.rc('font', size=fontsize)


def figure1(impc_data, impc_sd, impc_lb_italy, texfig=True, savefig=False):
    if texfig:
        set_texfig()

    plt.plot(impc_data, color='black', marker='o', label='Data from Fagereng et al. (2021)')
    plt.plot(impc_data + 1.96 * impc_sd, color='black', linestyle='--', linewidth=0.7)
    plt.plot(impc_data - 1.96 * impc_sd, color='black', linestyle='--', linewidth=0.7)
    plt.axhline(0, color='#808080', linestyle=':')

    plt.scatter(range(len(impc_data)), impc_lb_italy, marker='D', color='#9A0000', label='Lower bound from SHIW 2016')

    plt.xlabel(r"Year $t$")
    plt.ylabel(r"iMPC $M_{t, 0}$")
    plt.xlim([-0.1,len(impc_data)-1+0.1])
    plt.legend(framealpha=0)
    plt.tight_layout()
    if savefig:
        plt.savefig('figures/fig1.pdf', format='pdf', transparent=True)
    plt.show()


def figure2(impc_data, impc_sd, Ms, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)
    N = len(impc_data)

    # Figure 2a
    plt.figure(figsize =(6, 4.5))
    plt.scatter(range(N), impc_data, color='black', marker='o',s=50, label='Data')
    plt.plot(impc_data + 1.96 * impc_sd, color='black', linestyle='--', linewidth=0.5)
    plt.plot(impc_data - 1.96 * impc_sd, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='#808080', linestyle=':')

    plt.plot(Ms['HA-one'][:N, 0], color='blue', linestyle='dashed', label='HA-one', linewidth=2.5)
    plt.plot(Ms['TABU'][:N, 0], color='purple',  linestyle='dashdot', label='TABU / ZL', linewidth=2.5)
    plt.plot(Ms['HA-two'][:N, 0], color='green', label='HA-two', linewidth=2.5)

    plt.xlim([-0.1,N-1+0.1])
    plt.xlabel(r"Year $t$")
    plt.ylabel(r"iMPC $M_{t, 0}$")
    plt.legend(framealpha=0)
    plt.title(r'(a) Data and model fit')
    plt.tight_layout()
    if savefig:
        plt.savefig('figures/fig2_a.pdf', format='pdf', transparent=True,  bbox_inches = "tight")
    plt.show()

    # Figure 2b
    plt.figure(figsize =(6, 4.5))

    plt.scatter(range(N), impc_data, color='black', marker='o',s=50, label='Data')
    plt.plot(impc_data + 1.96 * impc_sd, color='black', linestyle='--', linewidth=0.5)
    plt.plot(impc_data - 1.96 * impc_sd, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=0, color='#808080', linestyle=':')

    plt.plot(Ms['RA'][:N, 0], color='red', label='RA',  linewidth=2.5)
    plt.plot(Ms['TA'][:N, 0], color='orange',  linestyle='dashed', label='TA', linewidth=2.5)
    plt.plot(Ms['BU'][:N, 0], color='gray', linestyle='dashdot', label='BU', linewidth=2.5)
    plt.plot(Ms['HA-hi-liq'][:N, 0], color='pink', linestyle='dotted', label='HA-hi-liq', linewidth=2.5)

    plt.xlim([-0.1,N-1+0.1])
    plt.xlabel(r"Year $t$")
    plt.ylabel(r"iMPC $M_{t, 0}$")
    plt.legend(framealpha=0)
    plt.title(r'(b) Alternative models')
    plt.tight_layout()
    if savefig: plt.savefig('figures/fig2_b.pdf', format='pdf', transparent=True,  bbox_inches = "tight")
    plt.show()


def figure3_and_D1(Ms, Mflag=True, texfig=True, savefig=False):
    """Plot M matrices in Figure 3 if M_flag=True, else plot A matrices (supplied in Ms) in Figure D1."""
    if texfig:
        set_texfig(fontsize=18.5)
 
    models = [('RA', 'HA-hi-liq'), 'TA', 'BU', ('TABU', 'ZL'), 'HA-one', 'HA-two']
    colors = [('red','pink'), 'orange', 'gray',  ('purple', 'brown'), 'blue', 'green']
    titles = ['(a) RA/HA-hi-liq', '(b) TA', '(c) BU',  '(d) TABU/ZL','(e) HA-one', '(f) HA-two']
    suffixes = ['a', 'b', 'c', 'd', 'e', 'f']
    linestyle = {'HA-hi-liq': 'dotted', 'ZL': 'dashed'}
    alphas= [1,0.85,0.7,0.55,0.4]
    columns = [0, 5, 10, 15, 20]
    Tplot = 35 if Mflag else 40

    for i, (ms, cs, title, suffix) in enumerate(zip(models, colors, titles, suffixes)):
        # make everything into iterables to handle case with just one model
        ms = [ms] if isinstance(ms, str) else ms
        cs = [cs] if isinstance(cs, str) else cs

        # iterate through columns and plot at different opacities alpha
        plt.title(title)
        for col, alpha in zip(columns, alphas):
            for m, c in zip(ms, cs):
                kwargs = {'linestyle': linestyle[m]}  if m in linestyle else {}
                plt.plot(Ms[m][0:Tplot, col], alpha=alpha, color=c, linewidth=2.5, **kwargs)
        
        # specific formatting for plotting of Ms vs. As
        line_labels = [f'$s={col:2.0f}$' for col in columns]
        if Mflag:
            # Ms case, Figure 3
            plt.yticks([0, 0.2, 0.4, 0.6])
            plt.ylim([0,0.6])
            if ms[0] == 'TA':
                plt.legend(line_labels, loc='upper right', borderpad=0.2)
        else:
            # As case, Figure D.1
            plt.axhline(y=0, color='#808080', linestyle=':')
            if ms[0] == 'RA':
                plt.ylim([-1,1])
            else:
                plt.ylim([-0.35,0.64])
            if ms[0] == 'BU':
                plt.legend(line_labels, loc='upper right', borderpad=0.2, prop={'size':17})

        # add lines to distinguish different models in same plot
        if len(ms) > 1:
            if not Mflag and ms[0] == 'RA':
                plt.legend(ms, frameon=False, loc= "upper left", bbox_to_anchor = (0.55, 0.32))
            else:
                plt.legend(ms, frameon=False, loc='upper right')
          
        if i>=3:
            plt.xlabel(r"Year $t$")
        if (i==0 or i==3):
            label = r"iMPC $M_{t, s}$" if Mflag else r"Asset Jacobian $A_{t, s}$"
            plt.ylabel(label)
        
        #plt.tight_layout()
        if savefig:
            fig_id = '3' if Mflag else 'D1'
            plt.savefig(f'figures/fig{fig_id}_{suffix}.pdf', format='pdf', transparent=True, bbox_inches = "tight")
        plt.show()


def figure4(Ms, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)

    # Figure 4(a), log first column
    plt.figure(figsize =(6, 4.5))
    Tplot = 31
    plt.plot(np.log10(Ms['HA-one'][:Tplot, 0]), color='blue', linestyle='dashed', label='HA-one', linewidth=2.5)
    plt.plot(np.log10(Ms['TABU'][:Tplot, 0]), color='purple',  linestyle='dashdot', label='TABU/ZL', linewidth=2.5)
    plt.plot(np.log10(Ms['HA-two'][:Tplot, 0]), color='green', label='HA-two', linewidth=2.5)
    plt.legend(framealpha=0)
    plt.xlabel(r"Year $t$")
    plt.ylabel(r"$\log_{10}(M_{t, 0})$")
    plt.xlim([-1,Tplot])
    plt.title('(a) Log first column')
    if savefig:
        plt.savefig('figures/fig4_a.pdf', format='pdf', transparent=True,  bbox_inches = "tight")
    plt.show()

    # Figure 4(b), log fiftieth column
    plt.figure(figsize =(6, 4.5))
    col, nxt = 49, 30
    ts = np.arange(col-nxt, col+nxt)
    l0, = plt.plot(ts, np.log10(Ms['HA-one'][ts, col]), color='blue', linestyle='dashed', label='HA-one', linewidth=2.5)
    l1, = plt.plot(ts, np.log10(Ms['TABU'][ts, col]), color='purple',  linestyle='dashdot', label='TABU', linewidth=2.5)
    l2, = plt.plot(ts, np.log10(Ms['ZL'][ts, col]), color='brown',  linestyle='dotted', label='ZL', linewidth=2.5)
    l3, = plt.plot(ts, np.log10(Ms['HA-two'][ts, col]), color='green', label='HA-two', linewidth=2.5)

    leg1 = plt.legend(handles=[l0, l3], loc='upper left',  framealpha=0)
    plt.legend(handles=[l1, l2], loc='upper right', framealpha=0)
    plt.gca().add_artist(leg1)
    plt.xlabel(r'Year $t$')
    plt.ylabel(r'$\log_{10}(M_{t, 49})$')
    plt.title('(b) Log fiftieth column');
    if savefig:
        plt.savefig('figures/fig4_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figureD2ab(ss, title, suffix, texfig=True, savefig=False):
    """Figure D.2(a)-(b): policy functions for one-asset models"""
    if texfig:
        set_texfig(fontsize=18.5)
        
    data = ss.internals['hh']
    a_grid, a, y, r = data['a_grid'], data['a'], data['y'], ss['r']

    plt.plot((1+r)*a_grid+y[0], a[0,:], label='Low income', linewidth=2.5)
    plt.plot((1+r)*a_grid+y[5], a[5,:], label='Medium income', linewidth=2.5)
    plt.plot((1+r)*a_grid+y[10], a[10,:], label='High income', linewidth=2.5)
    plt.plot(a_grid, a_grid, 'k--')
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.legend(framealpha=0)
    plt.xlabel(r'Cash on hand $ \varepsilon Z + (1+r)a_{-} $')
    plt.ylabel(r'Asset choice $a$')
    plt.title(title)
    plt.savefig('figures/figD2_'+suffix+'.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figureD3ae(ss, texfig=True, savefig=False):
    """Figure D.3(a)-(e): policy functions for two-asset model"""
    if texfig:
        set_texfig(fontsize=18.5)

    data = ss.internals['twoasset_calvo']
    a_grid, b_grid, a, b, c, y, ra, rb = data['a_grid'], data['b_grid'], data['a'], data['b'], data['c'], data['y'], ss['ra'], ss['rb']

    # a) Illiquid policy function
    plt.plot((1+rb)*b_grid+y[0], a[1,5,0,:], linewidth=2.5)
    plt.plot((1+rb)*b_grid+y[0], a[0,0,0,:], linewidth=2.5)
    plt.plot((1+rb)*b_grid+y[10], a[0,10,0,:], linewidth=2.5)
    plt.plot(b_grid, b_grid, 'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel(r'Liquid cash on hand $ \varepsilon Z + (1+r)(1-\zeta) a^{liq}_{-} $')
    plt.ylabel(r'Illiquid account choice $a^{illiq}$')
    plt.title('(a) Illiquid policy ($a^{illiq}_{-}=0$)')
    plt.savefig('figures/figD3_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    # b) Liquid policy function
    plt.plot((1+rb)*b_grid+y[0], b[1,0,0,:], linewidth=2.5)
    plt.plot((1+rb)*b_grid+y[0], b[0,0,0,:], linewidth=2.5)
    plt.plot((1+rb)*b_grid+y[10], b[0,10,0,:], linewidth=2.5)
    plt.plot(b_grid, b_grid, 'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel(r'Liquid cash on hand $ \varepsilon Z + (1+r)(1-\zeta) a^{liq}_{-} $')
    plt.ylabel(r'Liquid account choice $a^{liq}$')
    plt.title('(b) Liquid policy ($a^{illiq}_{-}=0$)')
    plt.savefig('figures/figD3_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    # c) Consumption policy
    plt.plot((1+rb)*b_grid+y[0], c[1,0,0,:], label='Low income\nnon-adjuster', linewidth=2.5)
    plt.plot((1+rb)*b_grid+y[0], c[0,0,0,:], label='Low income adjuster', linewidth=2.5)
    plt.plot((1+rb)*b_grid+y[10], c[0,10,0,:], label='High income adjuster', linewidth=2.5)
    plt.plot(b_grid, b_grid, 'k--')
    plt.xlim(0,20)
    plt.ylim(0,8)
    plt.legend(framealpha=0, loc = "lower left", bbox_to_anchor = (0.32, 0.45), prop = {'size': 17} )
    plt.xlabel(r'Liquid cash on hand $ \varepsilon Z + (1+r)(1-\zeta) a^{liq}_{-} $')
    plt.ylabel(r'Consumption choice $\tilde{c}$')
    plt.title('(c) Consumption policy ($a^{illiq}_{-}=0$)')
    plt.savefig('figures/figD3_c.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    # d) Illiquid asset policy
    plt.plot(a_grid, a[1,0,:,0], linewidth=2.5)
    plt.plot(a_grid, a[0,0,:,0], linewidth=2.5)
    plt.plot(a_grid, a[0,10,:,0], linewidth=2.5)
    plt.plot(a_grid, a_grid, 'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel(r'Assets in illiquid account $a^{illiq}_{-}$')
    plt.ylabel(r'Illiquid account choice $a^{illiq}$')
    plt.title(r'(d) Illiquid policy ($a^{liq}_{-}=0$)')
    plt.savefig('figures/figD3_d.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    # e) Liquid asset policy
    plt.plot(a_grid, b[0,0,:,0], label='adjuster, low income', color='tab:orange', linewidth=2.5)
    plt.plot(a_grid, b[0,10,:,0], label='adjuster, high income', color='tab:green', linewidth=2.5)
    plt.plot(a_grid, a_grid, 'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel(r'Assets in illiquid account $a^{illiq}_{-}$')
    plt.ylabel(r'Liquid account choice $a^{liq}$')
    plt.title('(e) Liquid policy ($a^{liq}_{-}=0$)')
    plt.savefig('figures/figD3_e.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def plot_two_dist(Ds, grids, newedges):
    """Sets up plot for two distributions in the style of figure D.2(c) and figure D.3(f)"""
    newpdfs, masscons, massaboves = {}, {}, {}
    for m, D in Ds.items():
        cumdist = D.cumsum()
        newcdf = interpolate(newedges, grids[m], cumdist)
        newpdfs[m] = newcdf[1:]-newcdf[:-1]
        masscons[m], massaboves[m] = cumdist[0], 1-newcdf[-1]

    ms = list(Ds)
    plt.hist(newedges[:-1], bins=newedges, weights=newpdfs[ms[0]], label=ms[0], linewidth=2.5);
    plt.hist(newedges[:-1], bins=newedges, weights=newpdfs[ms[1]], alpha=0.65, label=ms[1], linewidth=2.5);

    plt.annotate(f"Mass at constraint = {masscons[ms[1]]:.2f}", xy=(0, 0.085), xytext=(0.5, 0.08), color = 'tab:orange',
            arrowprops=dict(facecolor='tab:orange', edgecolor='tab:orange', width = 2, headwidth=7, shrink=0.05))
    plt.annotate(f"Mass above = {massaboves[ms[1]]:.2f}", xy=(5, 0.01), xytext=(2.5, 0.065), color = 'tab:orange',
            arrowprops=dict(facecolor='tab:orange', edgecolor='tab:orange', width = 2, headwidth=7, shrink=0.01))
    
    plt.annotate(f"Mass at constraint = {masscons[ms[0]]:.2f}", xy=(0, 0.01), xytext=(1, 0.04), color = 'tab:blue',
            arrowprops=dict(facecolor='k', edgecolor='k', width = 2, headwidth=7,shrink=0.05))
    plt.annotate(f"Mass above = {massaboves[ms[0]]:.2f}", xy=(5, 0.005), xytext=(2, 0.02), color = 'tab:blue',
            arrowprops=dict(facecolor='k', edgecolor='k', width = 2, headwidth=7, shrink=0.01))
    
    plt.xlim(0, newedges[-1])
    plt.legend(framealpha=0)


def figureD2c(ss_quant, texfig=True, savefig=False):
    """Figure D.2(c): wealth distributions for one-asset models"""
    if texfig:
        set_texfig(fontsize=18.5)

    newedges = np.linspace(0.001, 5, 40)
    models = ['HA-hi-liq', 'HA-one']
    Ds = {m: ss_quant[m].internals['hh']['D'].sum(axis=0) for m in models}
    grids = {m: ss_quant[m].internals['hh']['a_grid'] for m in models}

    plot_two_dist(Ds, grids, newedges)

    plt.xlabel('Assets $a$')
    plt.title('(c) Wealth distributions')
    plt.savefig('figures/figD2_dist.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figureD3f(ss, texfig=True, savefig=False):
    """Figure D.3(f): liquid and illiquid wealth distributions for two-asset model"""
    if texfig:
        set_texfig(fontsize=18.5)

    data = ss.internals['twoasset_calvo']
    D, a_grid, b_grid = data['D'], data['a_grid'], data['b_grid']

    newedges = np.linspace(0.001, 5, 40)
    Djoint = D.sum(axis=(0,1)) # sum across Calvo and income
    Ds = {'illiquid': Djoint.sum(axis=1), 'liquid': Djoint.sum(axis=0)}
    grids = {'illiquid': a_grid, 'liquid': b_grid}

    plot_two_dist(Ds, grids, newedges)

    plt.xlabel('Assets $a^{liq}, a^{illiq}$')
    plt.title('(f) Wealth distributions by account')
    plt.savefig('figures/figD3_f.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def gini_top10(D, a):
    """Calculate Gini and top 10% share from distribution D on grid of asset levels a"""
    Da = D*a
    cumdist = D.cumsum()
    cumdista = Da.cumsum() / Da.sum()
    cumdist, cumdista = np.concatenate(([0], cumdist)), np.concatenate(([0], cumdista))
    gini = 1-2*trapz(cumdista, cumdist)
    top10share = 1-interpolate([0.9], cumdist, cumdista)
    return gini, top10share[0]


def tableD1(ss_het):
    """Summary statistics for the heterogeneous-agent models"""
    htm_share, liq_share, whtm_share = {}, {}, {}
    htm_inc_share, liq_inc_share, whtm_inc_share = {}, {}, {}
    gini, top10share = {}, {}

    # first loop over the one-asset models
    for m in ('HA-hi-liq', 'HA-one'):
        data = ss_het[m].internals['hh']
        D, grid, a, y = data['D'], data['a_grid'], data['a'], data['y']
        Dy = D*y[:, np.newaxis]
        
        htm_share[m] = D[a==0].sum()
        liq_share[m] = D[a<=0.1].sum()
        htm_inc_share[m] = Dy[a==0].sum() / Dy.sum()
        liq_inc_share[m] = Dy[a<=0.1].sum() / Dy.sum()
        whtm_share[m], whtm_inc_share[m] = None, None
        gini[m], top10share[m] = gini_top10(D.sum(0), grid)

    # now do similar computation for the two-asset model
    m = 'HA-two'
    data = ss_het[m].internals['twoasset_calvo']
    D, a_grid, b_grid, a, b, y = data['D'], data['a_grid'], data['b_grid'], data['a'], data['b'], data['y']
    Dy = D*y[:, np.newaxis, np.newaxis]

    htm_share[m] = D[b==0].sum()
    liq_share[m] = D[b<=0.1].sum()
    whtm_share[m] = D[(b==0)*(a>0)].sum()
    htm_inc_share[m] = Dy[b==0].sum() / Dy.sum()
    liq_inc_share[m] = Dy[b<=0.1].sum() / Dy.sum()
    whtm_inc_share[m] = Dy[(b==0)*(a>0)].sum() / Dy.sum()

    # need to construct sorted total wealth distribution for two-asset model for gini/top 10
    wealth_grid = (a_grid[:,np.newaxis] + b_grid[np.newaxis,:]).ravel()
    Dwealth = D.sum(axis=(0,1)).ravel()
    ind = np.argsort(wealth_grid)
    wealth_grid, Dwealth = wealth_grid[ind], Dwealth[ind]
    gini[m], top10share[m] = gini_top10(Dwealth, wealth_grid)
    
    # make dict of Zs
    Z = {m: ss_het[m]['Z'] for m in gini}

    # now return a nice pandas dataframe
    df = pd.DataFrame([htm_share, liq_share, whtm_share, htm_inc_share, liq_inc_share, whtm_inc_share, gini, top10share, Z])
    df.index = ['Share of HtM (aliq = 0)', 'Share with liquid assets aliq <= 0.1', 'Share of WHtM (aliq = 0, ailliq>0)',
                'Share of income to HtM', 'Share of income to aliq <= 0.1', 'Share of income to WHtM',
                'Gini coefficient', 'Top 10pc share', 'Post-tax income']
    return df


def table2(params, ss_het, calib, calib_ha_one, calib_ha_two):
    models_het = ['HA-hi-liq', 'HA-one', 'HA-two']
    models = ['RA', 'HA-hi-liq', 'TA', 'BU', 'TABU', 'ZL', 'HA-one', 'HA-two']

    # build a dictionary for each row of the table
    r = calib['r']
    eis = {m: calib['eis'] for m in models}
    rs = {m: r for m in models}

    A_Z_target = ss_het['HA-two']['A']/ss_het['HA-two']['Z']
    A_Z = {m: A_Z_target for m in models}
    A_Z['HA-one'], A_Z['ZL'] = ss_het['HA-one']['A']/ss_het['HA-one']['Z'], 0

    beta = {m: ss_het[m]['beta'] for m in models_het}
    beta.update({m: params[m]['beta'] for m in ('BU', 'TABU', 'ZL')})
    beta['RA'] = beta['TA'] = 1/(1+calib['r'])

    # use actual hand-to-mouth for analytical models, fraction of income earned by constrained for quantitative
    mu = {m: params[m]['mu'] for m in ('TA', 'TABU', 'ZL')}
    mu['RA'], mu['BU'] = 0, 0
    for m in models_het:
        data = ss_het[m].internals['twoasset_calvo'] if m == 'HA-two' else ss_het[m].internals['hh']
        D, y, a = data['D'], data['y'], (data['b'] if m == 'HA-two' else data['a'])
        Dy = D*y[:, np.newaxis, np.newaxis] if m == 'HA-two' else D*y[:, np.newaxis]
        mu[m] = Dy[a==0].sum() / Dy.sum()

    # only for analytical models with BU-like component
    lamb = {m: params[m]['lamb'] for m in ('BU', 'TABU', 'ZL')}
    ms = {m: 1 - params[m]['lamb'] / (1+r) for m in ('BU', 'TABU', 'ZL')}

    # only for quantitative models, all use the same parameters, but theta already build into sd_e
    # need to adjust sd_e back to sd_e/(1-theta) to get pretax theta
    rho_e, sd_e, theta = calib_ha_one['rho_e'], calib_ha_one['sd_e'], calib_ha_one['theta']
    sd_e /= 1-theta

    rho_es = {m: rho_e for m in models_het}
    sd_es = {m: sd_e for m in models_het}
    thetas = {m: theta for m in models_het}

    # last three are just for the two-asset model
    Ailliq_Z = {'HA-two': ss_het['HA-two']['Ailliq']/ss_het['HA-two']['Z']}
    spread = {'HA-two': (1+r)*calib_ha_two['zeta']}
    nu = {'HA-two': params['HA-two']['nu']}

    # build into pandas dataframe
    df = pd.DataFrame([eis, rs, A_Z, beta, mu, lamb, ms, rho_es, sd_es, thetas, Ailliq_Z, spread, nu])
    df.index = ['eis', 'r',  'A/Z', 'beta', 'mu', 'lambda', 'm', 'rho_e', 'sd_e', 'theta', 'Ailliq/Z', 'spread', 'nu']

    # formatting dataframe for display
    round3 = (df.index == 'theta') | (df.index == 'nu')
    df.loc[~round3] = df.loc[~round3].round(2)
    df.loc[round3] = df.loc[round3].round(3)
    
    # convert everything to string
    df = df.fillna("").astype(str)

    return df
