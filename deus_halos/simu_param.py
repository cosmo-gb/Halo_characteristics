#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 2023/12/21

@author: guillaume
"""

'''
This file allows to set the cosmological and numerical parameters
of any of the DEUS-FUR simulations
'''




import numpy as np
import sys
class Simu_param():

    def __init__(self, cosmo: str, z: float,):
        if cosmo != "lcdm" and cosmo != "rpcdm" and cosmo != "wcdm":
            sys.exit("cosmo should be either lcdm or rpcdm or wcdm")
        if z not in [0, 0.5, 1, 1.5, 2]:
            sys.exit("z should be either 0, 0.5, 1, 1.5 or 2")
        if cosmo == "lcdm":
            if z == 0:
                # to pass from a length from DEUS simulation in a length in centimetre at this redshift
                self.UNIT_L = 0.277801161516035E+28
                # to pass from a time from DEUS simulation in a time in second at this redshift
                self.UNIT_T = 0.428844370639077E+18
            elif z == 0.5:
                self.UNIT_L = 0.185393317409458E+28
                self.UNIT_T = 0.190994011556267E+18
            elif z == 1:
                self.UNIT_L = 0.138918824824999E+28
                self.UNIT_T = 0.107239258054031E+18
            elif z == 1.5:
                self.UNIT_L = 0.111165264918421E+28
                self.UNIT_T = 0.686704373992292E+17
            else:
                self.UNIT_L = 0.927614649851624E+27
                self.UNIT_T = 0.478152902920744E+17
            # ratio of matter in the Universe at z ~ 0,
            self.OMEGA_MATTER_0 = 0.257299989461899
            # mass of one simu particle in Msun/h
            self.MASS_ONE_PARTICLE = 2.2617151991513 * 10**9
        elif cosmo == "rpcdm":
            if z == 0:
                self.UNIT_L = 0.278178734289182E+28
                self.UNIT_T = 0.430010888402685E+18
            elif z == 0.5:
                self.UNIT_L = 0.185178199587640E+28
                self.UNIT_T = 0.190551035730688E+18
            elif z == 1:
                self.UNIT_L = 0.138983769703220E+28
                self.UNIT_T = 0.107339550704166E+18
            elif z == 1.5:
                self.UNIT_L = 0.111331500163824E+28
                self.UNIT_T = 0.688759689073803E+17
            else:
                self.UNIT_L = 0.927002596709038E+27
                self.UNIT_T = 0.477522127117482E+17
            # ratio of matter in the Universe at z = 0
            self.OMEGA_MATTER_0 = 0.230000004172325
            # mass of one simu particle in Msun/h
            self.MASS_ONE_PARTICLE = 2.0217432046123 * 10**9
        else:  # wcdm case
            if z == 0:
                self.UNIT_L = 0.277774065550533E+28
                self.UNIT_T = 0.428760718104870E+18
            elif z == 0.5:
                self.UNIT_L = 0.185365616808401E+28
                self.UNIT_T = 0.190936940959115E+18
            elif z == 1:
                self.UNIT_L = 0.138926260154223E+28
                self.UNIT_T = 0.107250737859288E+18
            elif z == 1.5:
                self.UNIT_L = 0.111262067415843E+28
                self.UNIT_T = 0.687900856391014E+17
            else:
                self.UNIT_L = 0.926558564612419E+27
                self.UNIT_T = 0.477064772693956E+17
            # ratio of matter in the Universe at z ~ 0
            self.OMEGA_MATTER_0 = 0.2750000
            # mass of one simu particle in Msun/h
            self.MASS_ONE_PARTICLE = 2.417301666231 * 10**9
        # commun param
        # 1 pc in meters, from the PDG, fixed value, no uncertainty
        self.PC_TO_METER = 3.08567758149 * 10**16
        self.HUBBLE_CONSTANT_DIMENSIONLESS = 0.72
        self.HUBBLE_CONSTANT = 100  # Hubble constant in h km s**-1 Mpc**-1
        # newton constant in m**3 kg**-1 s**-2, from the PDG of 2016, the last 2 numbers are uncertain (31)
        self.G_NEWTON = 6.67408 * 10**(-11)
        self.HSML = 0.005  # hsml in Mpc/h, (hydro smoothing length)
        # equations:
        self.L_BOX = (self.UNIT_L * 10**(-2))/((self.PC_TO_METER /
                                                self.HUBBLE_CONSTANT_DIMENSIONLESS) * 10**6)
        self.REDSHIFT = 648/self.L_BOX - 1
        self.METER_TO_PC = 1/self.PC_TO_METER
        # Newton constant in m kg**-1 (km/s)**2
        self.G_NEWTON = self.G_NEWTON * 10**(-6)
        # G_Newton in (Mpc/h) * (mass_one_particle)**(-1) * (km/s)**2
        self.G_NEWTON = self.G_NEWTON * self.MASS_ONE_PARTICLE
        # ratio of dark energy in the univers at z ~ 0
        self.OMEGA_LAMBDA_0 = 1 - self.OMEGA_MATTER_0
        self.HUBBLE_PARAMETER_DIMENSIONLESS = np.sqrt(
            (self.OMEGA_LAMBDA_0 + self.OMEGA_MATTER_0 * (1 + self.REDSHIFT)**3))
        # formula of the Hubble expansion rate as a function of z in unit of (km/s)/(Mpc/h)
        self.HUBBLE_PARAMETER = self.HUBBLE_CONSTANT * \
            self.HUBBLE_PARAMETER_DIMENSIONLESS
        # ratio of matter in the Universe as a function of redshift
        self.OMEGA_MATTER_Z = self.OMEGA_MATTER_0 * \
            ((1 + self.REDSHIFT)**3) * \
            ((1/self.HUBBLE_PARAMETER_DIMENSIONLESS)**2)
        # critical density at redshift z in (mass_one_particle) / (Mpc/h)**3
        self.RHO_CRIT = 3 * self.HUBBLE_PARAMETER * \
            self.HUBBLE_PARAMETER/(8 * np.pi * self.G_NEWTON)
        self.RHO_CRIT_0 = 3 * self.HUBBLE_CONSTANT * \
            self.HUBBLE_CONSTANT/(8 * np.pi * self.G_NEWTON)
        self.DELTA_VIR_BRIAN_NORMAN = (18 * (np.pi)**2) + (82 * (self.OMEGA_MATTER_Z - 1)) - (
            39 * (self.OMEGA_MATTER_Z - 1)**2)  # delta vir of Brian and Norman 1997
        # rho vir of Brian and Norman, in (mass_one_particle) / (Mpc/h)**3
        self.RHO_VIR_BRIAN_NORMAN = self.DELTA_VIR_BRIAN_NORMAN * self.RHO_CRIT
        # put the velocity in km/s at this redshift
        self.V_REN = self.UNIT_L/(self.UNIT_T * 10**5)


sim_par_z_0 = Simu_param(cosmo="lcdm", z=0)

sim_par_z_05 = Simu_param(cosmo="lcdm", z=0.5)

# from redhsift, I can build l_box and unit_l
print(sim_par_z_0.REDSHIFT, sim_par_z_0.L_BOX - 648/(1+sim_par_z_0.REDSHIFT))
unitl = sim_par_z_0.L_BOX * 100 * sim_par_z_0.PC_TO_METER * \
    (10**6) / sim_par_z_0.HUBBLE_CONSTANT_DIMENSIONLESS
print(unitl - sim_par_z_0.UNIT_L)
# let's try to build the unit_t
print(sim_par_z_0.UNIT_T, sim_par_z_05.UNIT_T)


def time_universe(redshift, H0, Omega_lambda_0, Omega_matter_0):
    coef = (1/H0) * 2/(3 * np.sqrt(Omega_lambda_0))
    num_1 = Omega_lambda_0 * (1 + redshift)**(-3)
    num_2 = num_1 + Omega_matter_0
    num = np.sqrt(num_1) + np.sqrt(num_2)
    den = np.sqrt(Omega_matter_0)
    return coef * np.log(num/den)


t0 = time_universe(0, sim_par_z_0.HUBBLE_CONSTANT,
                   sim_par_z_0.OMEGA_LAMBDA_0, sim_par_z_0.OMEGA_MATTER_0)

ren_time_in_years = sim_par_z_0.PC_TO_METER * 10**3

print(t0*ren_time_in_years/0.72, sim_par_z_0.UNIT_T)
