import numpy as np  
import matplotlib.pyplot as plt

from .sec5_plots import table4
from .sec34_plots import set_texfig

def figure7(mcap, mlab, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)
    Tplot = 11

    linestyles = ['solid', 'dashed', 'dashdot', 'solid']
    colors = ['red', 'orange', 'purple', 'green']
    models = ['RA', 'TA', 'TABU', 'HA-two']

   # Panel a
    plt.figure(figsize =(6, 4.5))
    for m, col, style in zip(models, colors, linestyles):
        plt.plot(mcap[m][0:Tplot], color=col, linestyle = style,  linewidth=2.5, label=m)
    plt.legend(framealpha=0)
    plt.ylabel(r'$m^{cap}_t$')
    plt.xlabel(r'Year $t$')
    plt.title(r'(a) iMPCs out of capital gains $\mathbf{m}^{cap}$')
    if savefig:
        plt.savefig('figures/fig7_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    
   # Panel b
    plt.figure(figsize =(6, 4.5))
    for m, col, style in zip(models, colors, linestyles):
        plt.plot(mlab[m][0:Tplot], color=col, linestyle = style,  linewidth=2.5, label=m)
    plt.ylabel(r'$M_{t,0}$')
    plt.xlabel(r'Year $t$')
    plt.title(r'(b) iMPCs out of income $\mathbf{M}_{t,0}$')
    if savefig:
        plt.savefig('figures/fig7_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figure8(impulses, texfig=True, savefig=False):
    if texfig: 
        set_texfig(fontsize=12)
    
    models = ['HA-two', 'RA', 'TA']
    color_list = ["green", "red", "orange"]
    line_style = ['solid', 'dotted', 'dashed']
    Tplot = 11

    row1 = ['Y', 'C', 'I']
    row1_names = ['Output', 'Consumption', 'Investment']
    row2 = ['N', 'pi', 'r']
    row2_names = [r'Hours (\% of ss)', 'Inflation (bps)', 'Real rate (bps)']

    fig, axs = plt.subplots(2, 4, figsize=(12, 5.5))
    
    for (m, col, style) in zip(models, color_list, line_style):
        for i, (k, name) in enumerate(zip(row1, row1_names)):
            axs[0, i].plot(impulses[m][k][:Tplot], color=col, linestyle=style, label=m)
            axs[0, i].title.set_text(name)

        for i, (k, name) in enumerate(zip(row2, row2_names)):
            imp = 100*impulses[m][k] if k in ('pi', 'r') else impulses[m][k]
            axs[1, i].plot(imp[:Tplot], color=col, linestyle=style, label=m)
            axs[1, i].title.set_text(name)

    # now add shocks in final column
    axs[0, 3].plot(impulses[models[0]]['G'][:Tplot], color='black')
    axs[0, 3].title.set_text('Government spending')
    axs[1, 3].plot(impulses[models[0]]['B'][:Tplot], color='black')
    axs[1, 3].title.set_text(r'Gov. bonds (\% of $Y_{ss}$)')

    axs[0, 0].legend(framealpha=0)
    axs[0, 0].set_ylabel(r'\% of $Y_{ss}$')

    for j in range(4):
        axs[1, j].set_xlabel('Years')

    plt.tight_layout()
    if savefig:
        plt.savefig('figures/fig8.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    
    plt.show()



def figureG(impulses, color, suffix, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=12)

    alphas = [1.0, 0.8, 0.4, 0.2]
    Tplot = 11

    row1 = ['Y', 'C', 'I', 'G']
    row1_names = ['Output', 'Consumption', 'Investment', 'Government spending']
    row2 = ['N', 'pi', 'r', 'B']
    row2_names = [r'Hours (\% of ss)', 'Inflation (bps)', 'Real rate (bps)', r'Gov. bonds (\% of $Y_{ss}$)']

    fig, axs = plt.subplots(2, 4, figsize=(12, 5.5))

    for m, alpha in zip(list(impulses), alphas):
        for i, (k, name) in enumerate(zip(row1, row1_names)):
            axs[0, i].plot(impulses[m][k][:Tplot], alpha=alpha, color=(color if i < 3 else 'black'), label=m)
            axs[0, i].title.set_text(name)

        for i, (k, name) in enumerate(zip(row2, row2_names)):
            imp = 100*impulses[m][k] if k in ('pi', 'r') else impulses[m][k]
            axs[1, i].plot(imp[:Tplot], alpha=alpha, color=color, label=m)
            axs[1, i].title.set_text(name)

    axs[0, 0].legend(framealpha=0)
    axs[0, 0].set_ylabel(r'\% of $Y_{ss}$')

    for j in range(4):
        axs[1, j].set_xlabel('Years')

    plt.tight_layout()
    if savefig:
        plt.savefig(f'figures/figG{suffix}.pdf', format='pdf', transparent=True, bbox_inches = "tight")


def figureF1a(mcap, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)
    plt.figure(figsize=(6, 4.5))
    Tplot = 15

    plt.plot(mcap['HA-two'][:Tplot], color='green',  linestyle = 'solid', linewidth=2.5, label='Same portfolio in both accounts')
    plt.plot(mcap['HA-two-alt'][:Tplot], color='green', linestyle = 'dashed', linewidth=2.5, label='All equity in illiquid account')
    plt.legend(framealpha=0)
    plt.ylim(-0.01, 0.3499995)
    plt.ylabel(r'MPC')
    plt.xlabel(r'Year $t$')
    plt.title(r'(a) MPC out of capital gains')
    if savefig:
        plt.savefig('figures/figF1_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")


def figureF1b(dC_bb, dC_bb_alt, dC_df, dC_df_alt, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)
    plt.figure(figsize=(6, 4.5))
    Tplot = 15

    plt.plot(dC_bb[:Tplot], color='green', linestyle = 'solid',  linewidth=2.5, label=r'$\rho_B = 0$')
    plt.plot(dC_bb_alt[:Tplot], color='green', linestyle = 'dashed', linewidth=2.5)
    plt.plot(dC_df[:Tplot], color='green',  linestyle = 'solid', alpha=0.5, linewidth=2.5, label=r'$\rho_B = 0.93$')
    plt.plot(dC_df_alt[:Tplot], color='green',  linestyle='dashed',  alpha=0.5, linewidth=2.5)
    plt.legend(framealpha=0, ncol = 2)
    plt.ylabel(r'Consumption (\% of $Y_{ss}$)')
    plt.xlabel(r'Year $t$')
    plt.title(r'(b) Consumption response to $G$')
    if savefig:
        plt.savefig('figures/figF1_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")


def figure10(decomp, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)

    Tplot = 15
    styles = ['solid', 'dashed', 'dashdot', 'dotted']

    # left panel: two-asset
    plt.figure(figsize=(6, 4.5))
    for k, style in zip(decomp['HA-two'], styles):
        plt.plot(decomp['HA-two'][k][:Tplot], color='green', linestyle=style, linewidth=2.5, label=k)
    plt.legend(framealpha=0)
    plt.ylabel(r'Consumption (\% of $Y_{ss}$)')
    plt.xlabel(r'Year $t$')
    plt.title(r'(a) Two-account model')
    if savefig:
        plt.savefig('figures/fig10_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")

    # right panel: TABU
    plt.figure(figsize=(6, 4.5))
    for k, style in zip(decomp['TABU'], styles):
        plt.plot(decomp['TABU'][k][:Tplot], color='purple', linestyle=style, linewidth=2.5, label=k)
    plt.legend(framealpha=0, ncol = 2)
    plt.ylabel(r'Consumption (\% of $Y_{ss}$)')
    plt.xlabel(r'Year $t$')
    plt.title(r'(b) TABU model')
    if savefig:
        plt.savefig('figures/fig10_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")


def figure9(rho_Bs, mult_impact, mult_cumul, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)

    model_list = ['TABU', 'HA-two', 'RA', 'TA']
    color_list = ['purple', 'green', 'red', 'orange']
    style_list = ['dashdot', 'solid', 'solid', 'dashed']

    plt.figure(figsize =(6, 4.5))
    for m, col, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_impact[m], color=col, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$dY_0/dG_0$')
    plt.legend(framealpha=0)
    plt.ylim(-0.5,1.5)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(a) Impact multiplier', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/fig9_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize =(6, 4.5))
    for m, col, style in zip(model_list, color_list, style_list):
        plt.plot(rho_Bs, mult_cumul[m], color=col, linestyle=style, label=m, linewidth=2.5)
    plt.xlabel(r'Persistence of debt $\rho_B$')
    plt.ylabel(r'$\sum_t (1+r)^{-t}dY_t/\sum_t (1+r)^{-t}dG_t$')
    plt.ylim(-0.5,1.5)
    plt.xlim(0,rho_Bs[-1])
    plt.title('(b) Cumulative multiplier', x=0.5, y=1.02)
    if savefig:
        plt.savefig('figures/fig9_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    plt.show()


def figure11(dYs_rho_76, dYs_rho_93, texfig=True, savefig=False):
    if texfig:
        set_texfig(fontsize=15)

    Tplot = 11
    styles = ['solid', 'dashed', 'dotted']

    plt.figure(figsize =(6, 4.5))
    for k, style in zip(dYs_rho_76, styles):
        plt.plot(dYs_rho_76[k][:Tplot], color='green', linestyle = style, linewidth=2.5, label=k)
    plt.legend(framealpha=0)
    plt.ylabel(r'Output (\% of $Y_{ss}$)')
    plt.xlabel(r'Year $t$')
    plt.title(r'(a) $\rho_B = 0.76$')
    if savefig:
        plt.savefig('figures/fig11_a.pdf', format='pdf', transparent=True, bbox_inches = "tight")

    plt.figure(figsize =(6, 4.5))
    for k, style in zip(dYs_rho_93, styles):
        plt.plot(dYs_rho_93[k][:Tplot], color='green', linestyle = style, linewidth=2.5, label=k)
    plt.ylim(-1, 15)
    plt.ylabel(r'Output (\% of $Y_{ss}$)')
    plt.xlabel(r'Year $t$')
    plt.title(r'(b) $\rho_B = 0.93$')
    if savefig:
        plt.savefig('figures/fig11_b.pdf', format='pdf', transparent=True, bbox_inches = "tight")
    
    
def table3(calibration):
    parameters = {'Capital share': 'alpha', 'Gov spending to GDP': 'G',
                  'Debt to GDP': 'B', 'Capital to GDP': 'K',
                  'SS price markup': 'mup', 'SS wage markup': 'muw', 'Depreciation rate': 'delta',
                  'Frisch elasticity of labor supply': 'frisch', 'Investment elasticity to q': 'epsI',
                  'Price flexibility': 'kappap', 'Wage flexibility': 'kappaw',
                  'Taylor rule coefficient': 'phi_pi', 'Persistence of gov spending': 'rhoG', 'Persistence of debt': 'rhoB'}
    
    for name in parameters:
        digits = 3 if name == 'Capital share' else 2
        print(f"{(name+':'):38}    {np.round(calibration[parameters[name]], digits)}  ")
