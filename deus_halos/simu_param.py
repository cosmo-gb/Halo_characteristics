#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 2023/12/21

@author: guillaume
"""

'''
This file extract allows to set the cosmological and numerical parameters
of any of the DEUS-FUR simulations
'''


import numpy as np
import sys


class Simu_param():
    
    def __init__(self, cosmo: str, z: float,):
        if cosmo != "lcdm" and cosmo!= "rpcdm" and cosmo != "wcdm":
            sys.exit("cosmo should be either lcdm or rpcdm or wcdm")
        if cosmo == "lcdm":
            self.unit_l = 1
            self.unit_t = 1
            self.Omega_matter_0 = 1
            self.mass_one_particle = 1
        elif cosmo == "rpcdm":
            self.unit_l = 1
            self.unit_t = 1
            self.Omega_matter_0 = 1
            self.mass_one_particle = 1
        else:
            self.unit_l = 1
            self.unit_t = 1
            self.Omega_matter_0 = 1
            self.mass_one_particle = 1
        # commun param
        self.pc_to_meter = 3.08567758149 * 10**16 # 1 pc in meters, from the PDG, fixed value, no uncertainty
        self.h = 0.72
        self.H0 = 100 # Hubble constant in h km s**-1 Mpc**-1                 
        self.G_Newton = 6.67408 * 10**(-11) # newton constant in m**3 kg**-1 s**-2, from the PDG of 2016, the last 2 numbers are uncertain (31)
        self.hsml = 0.005 # hsml in Mpc/h, (hydro smoothing length ?)    
        # equations:
        self.L_box = (self.unit_l * 10**(-2))/((self.pc_to_meter/h) * 10**6)
        self.redshift = 648/self.L_box - 1 
        self.meter_to_pc = 1/self.pc_to_meter
        self.G_Newton = self.G_Newton * 10**(-6) # Newton constant in m kg**-1 (km/s)**2
        self.G_Newton = self.G_Newton * self.mass_one_particle # G_Newton in (Mpc/h) * (mass_one_particle)**(-1) * (km/s)**2
        self.Ez = np.sqrt((self.Omega_lambda_0 + self.Omega_matter_0 * (1 + self.redshift)**3))                                                      
        self.Hz = self.H0 * self.Ez # formula of the Hubble expansion rate as a function of z in unit of (km/s)/(Mpc/h)                                                                                                      
        self.Omega_matter_z = self.Omega_matter_0 * ((1 + self.redshift)**3) * ((1/self.Ez)**2) # ratio of matter in the Universe as a function of redshift                                                                                   
        self.rho_crit = 3 * self.Hz * self.Hz/(8 * np.pi * self.G_Newton) # critical density at redshift z in (mass_one_particle) / (Mpc/h)**3           
        self.rho_crit_0 = 3 * self.H0 * self.H0/(8 * np.pi * self.G_Newton)     
        self.delta_vir_Brian_Norman = (18 * (np.pi)**2) + (82 * (self.Omega_matter_z - 1)) - (39 * (self.Omega_matter_z - 1)**2) # delta vir of Brian and Norman 1997 
        self.rho_vir_Brian_Norman = self.delta_vir_Brian_Norman * self.rho_crit # rho vir of Brian and Norman, in (mass_one_particle) / (Mpc/h)**3
        self.v_ren = self.unit_l/(self.unit_t * 10**5) # put the velocity in km/s at this redshift
