#!/usr/bin/env python
# coding: utf-8

ENERGY_NORM = 1e6
DENSITY_NORM = 1e20
from scipy.constants import electron_volt as eV
keV = 1000 * eV
from scipy.constants import mu_0
from scipy.constants import epsilon_0
import scipy.constants as const

import jax.numpy as jnp
import numpy as np

from .geometry import volume_average

# ### Pressure and total stored energy
def pressure(density, temperature):
    r"""
    density in e20/m^3
    temperature in keV
    
    Returns (single-species) pressure in Pa
    """
    return density * temperature * keV * DENSITY_NORM

# The stored energy is 
# 
# $$ \frac{3}{2} \int p \; dV = V \frac{3}{2} \left< p \right>$$
# 
# where $\left<x\right>$ is the volume average of $x$.
def storedenergy(V, density_profile, temperature_profile):
    r"""
    Coefficient 3 is 3/2 * 2
    where 3/2 is from elementary kinetics and 2 is for ne = ni
    
    returns energy in MJ
    """
    p = pressure(density_profile, temperature_profile)
    return V * 3 * volume_average(p) / ENERGY_NORM

def pressure_volume_average(density_profile, temperature_profile):
    r"""2 is for ions + electrons
    """
    p = pressure(density_profile, temperature_profile)
    return 2 * volume_average(p) / ENERGY_NORM

# ### Magnetic energy density

# The magnetic energy density is 
# $$ \frac{B_t^2}{2 \mu_0}. $$
def magnetic_energy_density(b):
    r"""
    Field energy density in MJ/m^3
    
    b in T
    """
    return b**2 / (2 * mu_0) / ENERGY_NORM

def fusionpowers(V, density_profile, temperature_profile):
    r"""Zeroed out in this mid-sized stellarator version.
    
    Parameters
    ----------
    V in m³
    density_profile: array_like,
        in e20/m³
    temperature_profile: array_like,
        in keV
    
    Returns
    -------
    dict with three entries, which are
        fusion powers in MW
    """
   
    fus_pow_dens = 0 #fusion_power_density(density_profile, temperature_profile)
    total_fusion_power = 0 #V * volume_average(fus_pow_dens)
    
    alpha_power = 0.2 * total_fusion_power
    neutron_power = 0.8 * total_fusion_power
    return {"total_fusion_power / MW": total_fusion_power,
            "alpha_power / MW": alpha_power,
            "neutron_power / MW": neutron_power}


# ### Bremsstrahlung computation

# The local bremsstrahlung source density is 
# $$ 5.355 \cdot 10^{-37} Z_\text{eff}\,n_e^2\,T_e^{1/2} $$
def bremsstrahlung_density(n_e, t_e, Z_eff):
    r"""
    ne and te profiles in e20 m⁻³ and keV
    returns power in MW/m^3
    """
    C_b = 5.355e-37 * DENSITY_NORM**2 # Johner, Helios paper
    return C_b * Z_eff * n_e**2 * t_e**(1/2) / ENERGY_NORM

# The total bremsstrahlung is the integral over the whole profile.
# 
# $$ P_{br} = V \left<  5.355 \cdot 10^{-37} Z_\text{eff}\,n_e^2\,T_e^{1/2} \right>.$$
# 
# I may use $Z_\text{eff} = 2$ as a gross approximation.
def bremsstrahlung(V, density_profile, temperature_profile, Z_eff=2.0):
    r"""
    Parameters
    ----------
    V in m³
    density_profile: array_like,
        in e20/m³
    temperature_profile: array_like,
        in keV
    
    Returns
    -------
    float, 
        total bremsstrahlung power in MW
    """
    brems_pow_dens = bremsstrahlung_density(density_profile,
                                            t_e=temperature_profile,
                                            Z_eff=Z_eff)
    return V * volume_average(brems_pow_dens)


# ### Energy confinement time
def total_heating(p_alpha, p_aux):
    """Naive P_heat"""
    return p_alpha + p_aux

# $$\tau_{\text{ISS}_{04}} = 0.134 f_\text{ren} \left(\frac{a}{\mathrm{m}}\right)^{2.28} \, \left(\frac{R_0}{\mathrm{m}}\right)^{0.64} \, \left(\frac{\bar{n_e}}{10^{19}\text{m}^{-3}}\right)^{0.54} \left(\frac{B_t}{\mathrm{T}}\right)^{0.84} \, \iota_{2/3}^{0.41} \left(\frac{P_\text{heat}}{\mathrm{MW}}\right)^{-0.61}$$

# Note that I convert the density internally to the required e19 units.
def iss04_confinement_time(major_radius, minor_radius, line_averaged_density,
                           b_toroidal, iota_two_thirds, p_heat, f_ren=1):
    r"""  
    major radius "R0" in m
    minor radius "a" in m
    line_averaged_density "\bar{n}" in 10^20
    b_toroidal "B_t"
    iota_two_thirds "ι_{2/3}"
    p_heat in MW?
    """
    r0 = major_radius
    a = minor_radius
    ne = line_averaged_density * DENSITY_NORM / 1e19
    bt = b_toroidal
    ι23 = iota_two_thirds
    p_heat_mw = p_heat
    const = 0.134
    
    τEISSO4 = const * f_ren * a**(2.28) * r0**(0.64) * ne**(0.54) * bt**(0.84) * ι23**(0.41) * p_heat_mw**(-0.61)
    return τEISSO4

# ## Volume-averaged beta

# 
# 
# The volume-averaged beta is
# $$ \left<\beta\right> = \frac{\int p \; dV }{\int B_t^2 / (2 \mu_0) \; dV} .$$
def beta_volume_average(volume_averaged_pressure,
                        magnetic_energy_density):
    r"""Includes pressure of ions + electrons
    
    Parameters
    ----------
    volume_averaged_pressure: float
        J/m^3; but note this is different from the stored energy W/V
    magnetic_energy_density: float
        J/m^3
    
    Assumes magnetic energy is constant in space
    
    $$ \left<p\right>_V / \left( V (B^2 / (2\mu_0) )\right) $$
    
    """
    return volume_averaged_pressure / magnetic_energy_density


# # collisionality
def collisionality(central_temperature, central_density, major_radius, iota, ions=True):
    #nu approximation from Goldston
    if ions:
        #nuii
        nu = 1e-12*central_density/central_temperature**1.5
        mass = const.proton_mass
    else:
        #nuei and nuee
        nu = 5e-11*central_density/central_temperature**1.5
        mass = const.electron_mass
    vth = jnp.sqrt(const.elementary_charge * central_temperature/mass)
    nustar = nu * major_radius / (vth * iota)
    return nustar

def normalized_ion_gyroradius(central_temperature, central_density, minor_radius, B_t):
    mass = 2.*const.proton_mass #deuterium mass
    
    omegac = const.elementary_charge * B_t / mass 
    vth = jnp.sqrt(const.elementary_charge * central_temperature/mass)
    rhostar = vth/omegac/minor_radius
    return rhostar



# # Experiment metrics

# Plasma Frequency
# 
# $$ \omega_p = \sqrt{\frac{n_e e^2}{m_e \epsilon_0}}$$
# 
# Electron gyrofrequency
# 
# $$ \omega_\mathrm{gyro} = \frac{e B}{m_e}$$
def plasma_frequency(central_electron_density):
    r"""
    Returns ω_p in radians/second (not cycles per second)
    """
    ne = central_electron_density
    m_e = const.electron_mass
    e_charge = const.elementary_charge
    return jnp.sqrt(ne * e_charge**2 / (m_e * epsilon_0))


def electron_gyrofrequency(B):
    """
    Calculate the electron gyrofrequency.

    Parameters:
    B (float): Magnetic field strength in Tesla.

    Returns:
    float: Electron gyrofrequency in Hz.
    """
    e = const.e  # elementary charge in coulombs
    m_e = const.m_e  # mass of an electron in kg

    f_c = e * B / (2 * np.pi * m_e)
    return f_c

def neutral_beam_mean_free_path(NBI_energy, uniform_density):
    pass

def nbi_slowing_down_time(NBI_energy, uniform_density, uniform_temperature):
    pass



# ### Powerplant metrics

# For the computation of the recirculating power, I assume that the auxilliary heating systems have an efficiency $\eta_\text{aux} = 0.5$.
# 
# $P_\text{recirc}= P_\text{aux} / \eta_\text{aux}$.
# 
# The total heating for engineering purposes is $P_\text{aux}$.
def recirculating_power(p_aux):    
    η_aux = 0.5
    return jnp.where(p_aux <=0.0, 0.0, p_aux / η_aux)

# ### Energy loss due to finite energy confinement time

# The power lost is $$P_\text{conf. loss} = W / \tau$$
def confinement_loss(stored_energy, energy_confinement):
    return stored_energy/energy_confinement

def power_balance_residual(p_heat, p_bremss, p_conf_loss):
    """
    from Lion Eq 7,
    P^conf_loss + P_br + P_line + P_sync = f_α P_α + f_not-α P_not-α + P_aux
    
    dW/dt = f_α P_α + f_not-α P_not-α + P_aux - (P^conf_loss + P_br + P_line + P_sync)
    """
    return p_heat - p_conf_loss - p_bremss

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