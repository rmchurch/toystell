#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt; plt.ion()
import numpy as np

my_size = 18

plt.rcParams.update({'font.size': my_size})

# Set the font size for the x and y labels
plt.rcParams['axes.labelsize'] = my_size

# Set the font size for the tick labels
plt.rcParams['xtick.labelsize'] = my_size
plt.rcParams['ytick.labelsize'] = my_size

# Set the font size for the plot title
plt.rcParams['axes.titlesize'] = my_size

def darker_color(color, amount=0.5):
    """
    Lightens the given color by multiplying luminosity by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_1dscan(x,plasmas):
    fig, ax = plt.subplots(2,2, figsize=(8,8), sharex=False)
    plt.suptitle("5 keV, 1e20 density, R=6.0, A=5.5")
    ax[0,0].plot(x, [p['volume averaged β']*100 for p in plasmas])
    ax[0,0].set_title("volume average β / %")

    ax[1,0].plot(x, [p['relative_density'] for p in plasmas])
    ax[1,0].set_title("$n / n_\mathrm{Sudo}$ (should be less than 1)", usetex=True, size=13)

    for axx in ax.reshape(4,):
        axx.set_xlabel("$B / \mathrm{T}$", usetex=True)

    plt.tight_layout()


    ax[0,1].plot(x, [p['auxilliary power / MW'] for p in plasmas])
    ax[0,1].set_title("Heating power / MW")

    ax[1,1].plot(x, [p['electron gyrofrequency / GHz'] for p in plasmas])
    ax[1,1].set_title('electron gyrofrequency / GHz')


def plot_2dscan(x, y, plasmas, fparams):
    spatial, profile = fparams
    R0, a, B0, _, _ = spatial
    A = R0/a
    
    beta = np.zeros((x.size, y.size))
    paux = np.zeros((x.size, y.size))
    relden = np.zeros((x.size, y.size))
    taue = np.zeros((x.size, y.size))
    nustar = np.zeros((x.size, y.size))
    rhostar = np.zeros((x.size, y.size))
    for iin,n in enumerate(y):
        for iit,T in enumerate(x):
            plasma = plasmas[iin][iit]
            beta[iit,iin] = plasma['volume averaged β']
            paux[iit,iin] = plasma['auxilliary power / MW']
            relden[iit,iin] = plasma['relative_density']
            taue[iit,iin] = plasma['confinement_time / s']
            nustar[iit, iin] = plasma['collisionality']
            rhostar[iit, iin] = plasma['normalized_gyroradius']
            
    #CONSTRAINT 1:  n/nsudo > 1
    mask = relden > 1.0
    #CONSTRAINT 2: Paux < 15 MW
    mask = mask | (paux > 15.0)
    beta = np.ma.masked_where(mask, beta)
    paux = np.ma.masked_where(mask, paux)
    relden = np.ma.masked_where(mask, relden)
    taue = np.ma.masked_where(mask, taue)
    nustar = np.ma.masked_where(mask, nustar)
    rhostar = np.ma.masked_where(mask, rhostar)

    #find max point to print out
    maxiit, maxiin = np.argwhere(beta == np.ma.max(beta[~mask]))[0]
    
    fig, ax = plt.subplots(2,3, figsize=(12,8), sharex=False)
    plt.suptitle(("R=%0.1f, A=%0.1f, B=%0.1f T \n " + \
                  "n0 = %0.2e $m^{-3}$, T0 = %0.1f keV\n"
                 "$\\beta$ = %0.2f %%, $P_{aux}$ = %0.1f MW, $n/n_{sudo}$ = %0.2f, \n $\\tau_E$ = %0.3e, $\\nu^*$ = %0.3e, $1/\\rho^*$ = %0.3f") \
                 % (R0,A,B0, \
                    y[maxiin]*1e20, x[maxiit], \
                    100*beta[maxiit, maxiin],paux[maxiit, maxiin],relden[maxiit, maxiin],taue[maxiit, maxiin],nustar[maxiit, maxiin],1./rhostar[maxiit, maxiin]))
    c00 = ax[0,0].contourf(x, y, beta.T*100, 30)
    ax[0,0].set_title("volume average β / %")
    fig.colorbar(c00,ax=ax[0,0])
    ax[0,0].plot(x[maxiit],y[maxiin],'r*')
    
    c10 = ax[1,0].contourf(x, y, relden.T, levels=np.linspace(0,1,30))
    ax[1,0].set_title("$n / n_\mathrm{Sudo}$ (should be less than 1)", usetex=True, size=13)
    fig.colorbar(c10,ax=ax[1,0])
    ax[1,0].plot(x[maxiit],y[maxiin],'r*')
    
    for axx in ax.reshape(6,):
        axx.set_xlabel("$T_0 / \mathrm{keV}$", usetex=True)
    
    c02 = ax[0,2].contourf(x, y, nustar.T, 30)
    ax[0,2].set_title("collisionality $\\nu^*$", usetex=True)
    fig.colorbar(c02,ax=ax[0,2])
    ax[0,2].plot(x[maxiit],y[maxiin],'r*')
    
    plt.tight_layout()
    
    
    c01 = ax[0,1].contourf(x, y, paux.T, 30)
    ax[0,1].set_title("Heating power / MW")
    fig.colorbar(c01,ax=ax[0,1])
    ax[0,1].plot(x[maxiit],y[maxiin],'r*')
    
    #ax[1,1].plot(x, [p['electron gyrofrequency / GHz'] for p in plasmas])
    #ax[1,1].set_title('electron gyrofrequency / GHz')
    
    #ax[1,1].set_visible(False)
    ax[1,1].set_title(r"$\tau_{E}$ / s", size=16, usetex=True)
    c11 = ax[1,1].contourf(x, y, taue.T, 30)
    fig.colorbar(c11,ax=ax[1,1])
    ax[1,1].plot(x[maxiit],y[maxiin],'r*')
    
    c12 = ax[1,2].contourf(x, y, 1./rhostar.T, 30)
    ax[1,2].set_title("1/normalized gyroradius $1/\\rho^*$", usetex=True)
    fig.colorbar(c12,ax=ax[1,2])
    ax[1,2].plot(x[maxiit],y[maxiin],'r*')

