import matplotlib.pyplot as plt
import pandas as pd
from .sec34_plots import set_texfig

plt.rc('font', size=15)


def figure5(rho_Bs, mult_impact, mult_cumul, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)
        
    model_list = ['HA-one', 'TABU', 'HA-two', 'RA', 'TA']
    color_list = ['blue', 'purple',  'green', 'red', 'orange']
    style_list = ['dashed', 'dashdot',  'solid', 'solid', 'dashed']

    # Impact multiplier
    plt.figure(figsize =(6, 4.5))
    for m, color, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_impact[m], color=color, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$dY_0/dG_0$')
    plt.legend(framealpha=0)
    plt.ylim(0.8,4)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(a) Impact multiplier', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/fig5_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    # Cumulative multiplier
    plt.figure(figsize =(6, 4.5))
    for m, color, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_cumul[m], color=color, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$\sum_t (1+r)^{-t}dY_t/\sum_t (1+r)^{-t}dG_t$')
    plt.ylim(0.8,4)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(b) Cumulative multiplier', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/fig5_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figureE2(rho_Bs, mult_impact, mult_cumul, texfig=True, savefig=False):
    """Same as Figure 5, but for all models, separating into "fitting" and other models"""
    if texfig:
        set_texfig(fontsize=15)
    
    model_list = ['HA-one', 'TABU', 'ZL', 'HA-two']
    color_list = ['blue', 'purple', 'brown', 'green']
    style_list = ['dashed', 'dashdot', 'dotted', 'solid']

    plt.figure(figsize =(6, 4.5))
    for m, color, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_impact[m], color=color, linestyle=style, label=m, linewidth=2.5)
    
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$dY_0/dG_0$')
    plt.legend(framealpha=0)
    plt.ylim(0.8,4)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(a) Impact multiplier - fitting models', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/figE2_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize =(6, 4.5))
    for m, color, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_cumul[m], color=color, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$\sum_t (1+r)^{-t}dY_t/\sum_t (1+r)^{-t}dG_t$')
    plt.legend(framealpha=0)
    plt.ylim(0.8,4)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(c) Cumulative multiplier - fitting models', x=0.5, y=1.02)
    if savefig: plt.savefig('figures/figE2_c.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    model_list = ['RA', 'TA', 'BU', 'HA-hi-liq']
    color_list = ['red', 'orange', 'gray', 'pink']
    style_list = ['solid', 'dashed', 'dashdot', 'dotted']

    plt.figure(figsize =(6, 4.5))
    for m, color, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_impact[m], color=color, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$dY_0/dG_0$')
    plt.legend(framealpha=0)
    plt.ylim(0.8,4)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(b) Impact multiplier - alternative models', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/figE2_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize =(6, 4.5))
    for m, color, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_cumul[m], color=color, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$\sum_t (1+r)^{-t}dY_t/\sum_t (1+r)^{-t}dG_t$')
    plt.legend(framealpha=0)
    plt.ylim(0.8,4)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(d) Cumulative multiplier  - alternative models', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/figE2_d.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figure6_E5(deltas, mult_disc, title, xlabel, id, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)
    
    model_list = ['HA-one', 'TABU',  'HA-two']
    color_list = ['blue', 'purple',  'green']
    style_list = ['dashed',  'dashdot',  'solid']

    plt.figure(figsize =(6, 4.5))
    for i, m in enumerate(model_list):
        plt.plot(deltas, mult_disc[m], color=color_list[i], linestyle=style_list[i], label=m, linewidth=2.5)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\sum_t (1+r)^{-t}dY_t/\sum_t (1+r)^{-t}dG_t$')
    plt.legend(framealpha=0)
    plt.ylim(1.5,8)
    plt.xlim(deltas[0],deltas[-1])
    plt.title(title, x=0.5, y=1.02)
    if savefig:
        plt.savefig(f'figures/fig{id}.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def table4(mult_impact, mult_cumul, ikc=True):
    """Create table 4 for either the IKC or quantitative environment"""
    # note that Table 4 looks at multipliers for the lowest and highest rhoB in grid
    # i.e. rhoB = 0 (balanced budget) and rhoB = 0.93 (highly deficit financed)
    models = ['RA', 'TA', 'TABU', 'HA-one', 'HA-two'] if ikc else ['RA', 'TA', 'TABU', 'HA-two']
    bb_impact = {m: mult_impact[m][0] for m in models}
    bb_cumul = {m: mult_cumul[m][0] for m in models}
    df_impact = {m: mult_impact[m][-1] for m in models}
    df_cumul = {m: mult_cumul[m][-1] for m in models}
    
    df = pd.DataFrame([bb_impact, bb_cumul, df_impact, df_cumul],
                      index=['BB impact', 'BB cumulative', 'DF impact', 'DF cumulative'])

    return df


def figureE1(dY_bench, dY_lumpsum, dY_redist, title, id, yticks=None, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)

    Tplot = 11
    plt.figure(figsize =(6, 4.5))
    plt.plot(dY_bench[:Tplot], color='blue', linestyle='solid', label='Benchmark', linewidth=2.5)
    plt.plot(dY_lumpsum[:Tplot], color='blue', linestyle='dashed', label='Lump-sum', linewidth=2.5)
    plt.plot(dY_redist[:Tplot], color='blue', linestyle='dashdot', label='Redistribution', linewidth=2.5)
    plt.xlabel(r'Year $t$')
    plt.ylabel(r'\% of $Y_{ss}$')
    if yticks is not None:
        plt.yticks(yticks)
    plt.legend(framealpha=0)
    plt.title(title, x=0.5, y=1.02)
    if savefig:
        plt.savefig(f'figures/figE1_{id}.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figureA2(dY_deficit, dY_monetary, dY_delev, title, id, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)

    Tplot=11
    plt.figure(figsize =(6, 4.5))
    plt.plot(dY_deficit[:Tplot], color='blue', linestyle='solid', label='Deficit-financed $G$', linewidth=2.5)
    plt.plot(dY_monetary[:Tplot], color='blue', linestyle='dashed', label='Monetary policy', linewidth=2.5)
    plt.plot(dY_delev[:Tplot], color='blue', linestyle='dashdot', label='Deleveraging', linewidth=2.5)
    plt.xlabel(r'Year $t$')
    plt.ylabel(r'\% of $Y_{ss}$')
    plt.legend(framealpha=0)
    plt.axhline(0, color='#808080', linestyle=':')
    plt.title(title, x=0.5, y=1.02)
    if savefig:
        plt.savefig(f'figures/figA2_{id}.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()
