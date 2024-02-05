#!/usr/bin/env python
# coding: utf-8

import numpy as np
import jax.numpy as jnp

# ## Construction of geometric volumes
# 
# The plasma is approximated as a circular torus, with major radius $R_0$ and minor radius $a$.
# 
# The plasma volume is
# 
# $$ V = 2 \pi^2 R_0 a^2. $$
#
def plasma_volume(major_radius, minor_radius):
    π = np.pi
    return 2 * π**2 * major_radius * minor_radius**2

def plasma_surface_area(major_radius, minor_radius):
    π = np.pi
    R0 = major_radius
    a = minor_radius
    return (2 * π * R0) * (2 * π * a)

# Fusion rates, bremsstahlung, volume-averaged beta, and line-averaged density & temperature are computed using a radial grid. Values are evaluated at the center of each grid cell.
def centered_grid(n):
    r"""In n intervals from 0 to 1
    """
    Δx = 1 / n
    return jnp.linspace(Δx / 2, 1 - Δx / 2, n)

def parab_profile(x0, αx, ρ):
    return x0 * (1 - ρ**2)**αx

def metric_coefficient(rho):
    r"""\sqrt(g)
    
    Circular stellarator approximation.
    I'm not 100% sure this is correct to call this a "metric coefficient"
    
    """
    return rho

def line_average(x):
    return np.mean(x)

def volume_average(x):
    r""" x is centered in intervals
    
        2 π R a^2 \int(2 π y(x) x dx)
    Y = ----------------------------- = 2 \int(y(x) x dx)
               2 π^2 R a^2
    """
    n_cells = len(x)
    
    ρ = centered_grid(n_cells)
    Δρ = 1 / n_cells
    return 2 * jnp.sum(x * metric_coefficient(ρ)) * Δρ



# ### Sudo density limit

# The Sudo "density limit" is 
# $$n_\text{Sudo} = 0.25 \left( \frac{P_\text{heat} B_t}{R_0 a^2} \right)^{0.5}$$
def sudo_density_limit(major_radius, 
                       minor_radius,
                       total_heating_power,
                       toroidal_field,
                       ):
    r"""
    Parameters
    ----------
    major radius: float, m
    minor radius: float, m
    total_heating_power: float, MW
    toroidal_field: float, T
    
    Returns
    -------
    Density in 1e20
    """
    const = 0.25
    r = major_radius
    a = minor_radius
    p = total_heating_power # MW
    b = toroidal_field
    return const * (p * b / (r * a**2))**0.5