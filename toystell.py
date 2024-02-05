#!/usr/bin/env python
# coding: utf-8

#toy (zero-to-one-dimensional) model of a stellarator. While the model is very simple, the results it produces are hopefully (at least for reasonable inputs) on the correct order of magnitude. The behavior of the model is not quite trivial: it has enough interesting features that can serve as food for thought for more complicated models, while being easy to work with numerically.
# 
# The model has just three "physics" elements:
# * Energy input: Auxilliary power
# * Energy loss: ISS04 confinement time scaling
# * Energy loss: Bremsstrahlung radiation
# 
# For a given set of inputs, the auxilliary power is solved for so that the total plasma stored energy $W$ is "stationary", that is, $dW/dt = 0$.
# 
# The plasma has the geometry of a circular torus, with major and minor radii $R_0$, $a$. There is a constant toroidal magnetic field $B_t$. There is a rotational transform $\iota$ defined at $2/3$ of the minor radius, $\iota_{2/3}$.
# The geometry is that of a particular stellarator, with a certain volume, surface area, dV/ds, iota profile, etc.
# 
# The density and temperature profiles are prescribed:
# 
# $$
# \begin{aligned}
# n(\rho) &= n_0\;(1-\rho^2)^{\alpha_n}\\
# T(\rho) &= T_0\;(1-\rho^2)^{\alpha_T}.
# \end{aligned}
# $$
# 
# Here $T_i = T_e$ and there are no impurities, so $$n_e = n_i = n_D + n_T$$.
# 
# Based on the profiles, it's easy to calculate bremsstrahlung $P_{br}$.
# I use the nonrelativistic formula for bremsstrahlung. This is integrated over the volume of the plasma.
# 
# The ISS04 confinement time scaling formula takes, as one of its inputs, the total heating power.
# This is the sum of the alpha power $P_{\alpha} = \frac{1}{5} P_{fus}$ and the auxilliary power $P_{aux}$.
# 
# The equation for the time-dependent stored energy is
# 
# $$ \frac{dW}{dt} = P_{aux} - \frac{W}{\tau_\text{ISS04}} - P_{br} $$
# 
# For each combination of "spatial" inputs $(R_0, a, B_t, \iota_{2/3}, f_\text{ren})$ and "profile" inputs $(T_0, n_0, \alpha_T, \alpha_n, Z_\text{eff})$ there is one solution for the required $P_{aux}$ which admits a stationary plasma, $dW/dt = 0$.
# 
# I solve for this $P_{aux}$ using a simple fixed-point iteration method.
# 
# The required $P_{aux}$ is positive for a system that produces no fusion power.
# * If positive, we can imagine that the plasma is being heated by RF power, neutral beams, or other generic means.

from copy import copy, deepcopy

from icecream import ic

import numpy as np

import functools

from scipy.optimize import minimize
from scipy.optimize import root
from scipy.optimize import root_scalar

from scipy.optimize import Bounds

import jax
import jax.numpy as jnp
from jax import vmap, grad, jvp, vjp, jit

from jaxopt import FixedPointIteration
from jaxopt import GradientDescent
from jaxopt import GaussNewton, ScipyRootFinding

from .plot import plot_1dscan, plot_2dscan
from .geometry import *
from .plasma import *

ENERGY_NORM = 1e6
DENSITY_NORM = 1e20



class ToyStellarator():

    def __init__(self, R0=12.0, a=2.8, B_t=5.0, iota_two_thirds=0.2, f_ren=1.0, 
                       T0 = 15.0, n0=0.8, alphaT = 2.0, alphan=2.0, Z_eff=2.0):
        spatial_params = {"R0": R0, "a":a, "B_t": B_t, "ι_2/3": iota_two_thirds, "f_ren": f_ren}
        profile_params = {"n0": n0, "αn": alphan, "αT": alphaT, "T0": T0, "Z_eff": Z_eff}
        params = {'spatial': spatial_params, 'profile': profile_params}
        self.fparams = self.make_fparams(params)

        self.paux_dT = jit(jax.grad(ToyStellarator.stationary_auxpower, argnums=1))


    # ### Utility functions for describing and modifying the design point

    # These functions translate back and forth between dictionary representations of the machine's "spatial" configuration and the plasma "profile" state, and tuple configurations. 
    # The dictionaries are easier to read and interact with, but for technical reasons, it's easier to use tuples with `jax`.

    def make_params(self,fparams):
        sp, pr = fparams
        spatial = {'R0': sp[0], 'a': sp[1], 'B_t': sp[2], 'ι_2/3': sp[3], 'f_ren': sp[4]}
        profile = {'T0': pr[0], 'n0': pr[1], 'αT': pr[2], 'αn': pr[3], 'Z_eff': pr[4]}
        return {'spatial': spatial, 'profile': profile}

    def make_fparams(self, params):
        s = params['spatial']
        p = params['profile']
        sp = s["R0"], s["a"], s["B_t"], s["ι_2/3"], s["f_ren"]
        pr = p["T0"], p["n0"], p["αT"], p["αn"], p["Z_eff"]
        return sp, pr

    # These functions modify the profile and spatial configuration, respectively, of a design point.
    # The second always sets the aspect ratio to 7.
    def set_T0n0(self, T0, n0, fparams):
        sp, pr = fparams
        new_pr = T0, n0, pr[2], pr[3], pr[4]
        return sp, new_pr

    def set_R0ABt(self, R0, A, B_t, fparams):
        sp, pr = fparams
        new_sp = R0, R0/A, B_t, sp[3], sp[4]
        return new_sp, pr

    def set_zeff(self, z_eff, fparams):
        sp, pr = fparams
        new_pr = pr[0], pr[1], pr[2], pr[3], z_eff
        return sp, new_pr

    def set_fren(self, f_ren, fparams):
        sp, pr = fparams
        new_sp = sp[0], sp[1], sp[2], sp[3], f_ren
        return new_sp, pr

    def set_iota(self, iota, fparams):
        sp, pr = fparams
        new_sp = sp[0], sp[1], sp[2], iota, sp[4]
        return new_sp, pr



    @staticmethod
    def plasma_moments(fparams):
        #def plasma_moments(spatial_params, n0, T0, αn, αT):
        #if fparams is None: fparams = self.fparams
        spatial, profile = fparams
        major_radius, minor_radius, B_t, _, _ = spatial
        central_temperature, central_density, temperature_profile_exponent, density_profile_exponent, z_eff = profile
        
        ρs = centered_grid(100)
        
        ns = parab_profile(central_density, density_profile_exponent, ρs)
        ts = parab_profile(central_temperature, temperature_profile_exponent, ρs)
        
        # prevent strict zeros
        ts = jnp.maximum(ts, 1e-3)
        
        V_plasma = plasma_volume(major_radius, minor_radius)
        SA_plasma = plasma_surface_area(major_radius, minor_radius)
        
        fusion_products = fusionpowers(V_plasma, ns, ts)
        neutron_power = fusion_products["neutron_power / MW"]
        
        neutron_wall_load = neutron_power / SA_plasma
        
        p_brems = bremsstrahlung(V_plasma, ns, ts, z_eff)
        
        n_e_bar = line_average(ns)
        T_bar = line_average(ts)
        
        n_avg = volume_average(ns)
        T_avg = volume_average(ts)

        W_stored = storedenergy(V_plasma, ns, ts)
        pressure_avg = pressure_volume_average(ns, ts)
        mag_energy_dens = magnetic_energy_density(B_t)
        
        volume_averaged_beta = beta_volume_average(pressure_avg, mag_energy_dens)
        
        return fusion_products | {'bremsstrahlung_power / MW': p_brems,
                                'line_average_electron_density': n_e_bar,
                                'line_average_temperature': T_bar,                              
                                'volume_average_electron_density': n_avg,
                                'volume_average_temperature': T_avg,                        
                                'neutron wall load / (MW/m^2)': neutron_wall_load,
                                'W / MJ': W_stored,
                                'volume averaged β': volume_averaged_beta}




    # ## Plasma and plant metrics for a design point with arbitrary auxilliary power

    # We have not yet solved for the required "steady" auxilliary power.
    # 
    # However, this function does compute the power balance residual $P_\text{residual}$.
    # 
    # Strictly speaking, it's not yet needed to compute all the power plant metrics, but from an organizational perspective it's easy to do that here.
    @staticmethod
    @jit
    def compute_plasma_point(fparams=None, p_aux=0):
        #if fparams is None: fparams = self.fparams
        spatial, profile = fparams
        R0, a, B_t, iota_two_thirds, f_ren = spatial
                
        plasma_integrals = ToyStellarator.plasma_moments(fparams)
        p_heat = total_heating(plasma_integrals["alpha_power / MW"], p_aux)
        ne_bar = plasma_integrals["line_average_electron_density"]
        
        τ_e = iss04_confinement_time(R0, a,
                                    line_averaged_density=ne_bar, b_toroidal=B_t,
                                    iota_two_thirds=iota_two_thirds, p_heat=p_heat, f_ren=f_ren)
        
        p_loss = confinement_loss(plasma_integrals["W / MJ"], τ_e)
        power_residual = power_balance_residual(p_heat, plasma_integrals["bremsstrahlung_power / MW"], p_loss)
        
        total_fusion_power = plasma_integrals["total_fusion_power / MW"]

        ω_plasma = plasma_frequency(profile[1]*1e20)
        f_gyro_e = electron_gyrofrequency(B_t) / 1e9
        
        n_c_sudo = sudo_density_limit(major_radius=R0,
                                    minor_radius=a,
                                    total_heating_power=p_heat,
                                    toroidal_field=B_t)
            
        relative_density = ne_bar / n_c_sudo

        #print(profile[0], profile[1], R0, iota_two_thirds)
        nustar = collisionality(profile[0]*1e3, profile[1]*1e20, R0, iota_two_thirds)
        rhostar = normalized_ion_gyroradius(profile[0]*1e3, profile[1]*1e20, a, B_t)
        
        return plasma_integrals | {'confinement_time / s': τ_e,
                                "total_plasma_heating / MW": p_heat,
                                "power_residual / MW": power_residual,
                                "sudo density / 1e20": n_c_sudo,
                                "relative_density": relative_density,
                                "auxilliary power / MW": p_aux,
                                "ω_plasma / (rad/s)": ω_plasma,
                                "electron gyrofrequency / GHz": f_gyro_e,
                                "collisionality": nustar,
                                "normalized_gyroradius": rhostar
                                }


    # same as compute_plasma_point, but only returns the power residual. Could be faster...
    @staticmethod
    @jit
    def _fast_power_residual(fparams, p_aux=0):
        spatial, profile = fparams
        R0, a, B_t, iota_two_thirds, f_ren = spatial
                
        plasma_integrals = ToyStellarator.plasma_moments(fparams)
        p_heat = total_heating(plasma_integrals["alpha_power / MW"], p_aux)
        ne_bar = plasma_integrals["line_average_electron_density"]
        
        τ_e = iss04_confinement_time(R0, a,
                                    line_averaged_density=ne_bar, b_toroidal=B_t,
                                    iota_two_thirds=iota_two_thirds, p_heat=p_heat, f_ren=f_ren)
        
        p_loss = confinement_loss(plasma_integrals["W / MJ"], τ_e)
        return power_balance_residual(p_heat, plasma_integrals["bremsstrahlung_power / MW"], p_loss)

    # ## The power balance residual and its derivatives

    # Convenience function to get the power balance residual.
    @staticmethod
    @jit
    def power_residual(spatial, T0, n0, αT, αn, z_eff, auxilliary_heating_power=0):
        profile = T0, n0, αT, αn, z_eff
        fparams = (spatial, profile)
        return ToyStellarator._fast_power_residual(fparams, auxilliary_heating_power)

    # ## Finding the stationary auxilliary power at any point (T0, n0).

    # For any point (T0, n0), the "stationary auxilliary power" is the amount needed so that the power balance residual $P_\text{residual} = 0$. There is exactly one auxilliary power level that satisfies $P_\text{residual} = 0$. It may be positive, zero, or negative.
    # 
    # (In contrast, for a given $(P_\text{aux}, n_0)$ there may be 0, 1, 2, or 3 positive real temperatures $T_0$ that lead to a zero power residual.)
    # 
    # I use a fixed point iteration method to find the stationary auxilliary power.

    @staticmethod
    def stationary_auxpower(spatial, T0, n0, αT, αn, z_eff):
        """Stationary aux power in MW"""
        def power_residual_fixedpoint_f(x, *args):
            residual = ToyStellarator.power_residual(*args, auxilliary_heating_power=x)
            return x - residual
        
        fpi = FixedPointIteration(fixed_point_fun=power_residual_fixedpoint_f, tol=1e-6)
        y_init= jnp.array([5.])
        return fpi.run(y_init, spatial, T0, n0, αT, αn, z_eff).params[0]
    
    @staticmethod
    @jit #TODO: This JUT causes TypeError: Cannot interpret value of type ToyStellataor
    def stationary_plasma_point(fparams):
        """Find the plasma state with the auxilliary power required s.t. dW/dt = 0
        
        Takes about 50us
        """
        #if not fparams: fparams = self.fparams
        spatial, profile = fparams
        p_aux_stationary = ToyStellarator.stationary_auxpower(spatial, *profile)
        plasma_point = ToyStellarator.compute_plasma_point(fparams, p_aux_stationary)
        return plasma_point

    # # Find plasma, machine stationary state for a given (n0, T0)

    # This function returns the plasma and machine state for a given "spatial" configuration and a given "profile" configuration, at the required $P_\text{aux}$.
    def plasma_at_T0n0(self, T0, n0, fparams):
        """Stationary plasma at this point
        
        Parameters
        ----------
        T0: float
            keV
        n0: float
            10^20 / m^3
        fparams: tuple of tuples
            ((R0, a, B_t, ι_2/3), (_, _, αT, αn))
            
        Returns
        -------
        dict
        
        Takes about 45us
        """    
        outputs = {}
            
        fparams = self.set_T0n0(T0, n0, fparams)
        
        plasma_point = ToyStellarator.stationary_plasma_point(fparams)
        
        spatial, profile = fparams
        dpaux_dt = self.paux_dT(spatial, *profile)
        outputs["dauxilliary_power/dT0 / (MW/keV)"] = dpaux_dt
        
        selected_outputs = ["bremsstrahlung_power / MW",
                            "volume averaged β",
                            "volume_average_electron_density",
                            "volume_average_temperature",
                            "sudo density / 1e20",
                            "relative_density",
                            "confinement_time / s",
                            "auxilliary power / MW",
                            "ω_plasma / (rad/s)",
                            "electron gyrofrequency / GHz",
                            "collisionality",
                            "normalized_gyroradius"]
        
        for ao in selected_outputs:
            if ao in plasma_point:
                outputs[ao] = plasma_point[ao]
        
        return outputs
    
    def plot_Bscan(self, b_fields=np.linspace(1.5,6.0)):
        plasmas = [self.plasma_at_T0n0(5.0, 1.0, self.set_R0ABt(6.0, 5.5, bb, self.fparams)) for bb in b_fields]
        plot_1dscan(b_fields, plasmas)


    def plot_nTscan(self, T_scan=np.linspace(1,10,50), n_scan=np.logspace(-2,0.5,40)):
        plasmas = []
        for iin,n in enumerate(n_scan):
            row = []
            for iit,T in enumerate(T_scan):
                row.append(self.plasma_at_T0n0(T, n, self.fparams))
            plasmas.append(row)
        plot_2dscan(T_scan, n_scan, plasmas, self.fparams)


