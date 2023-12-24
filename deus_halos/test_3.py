#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:03:30 2022

@author: guillaume
"""

import numpy as np

'''Some parameters of the simulation  '''
# see http://www.deus-consortium.org/deus-data/snapshots-data/#snapshots
##########################################################################
# simu DEUS-FUR, Lbox = 648 Mpc/h, rpcdmw7, z = 2, RPCDM DE
# see http://www.deus-consortium.org/deus-data/snapshots-data/#snapshots
# to pass from a length from DEUS simulation in a length in centimetre at this redshift
unit_l = 0.927002596709038E+27
# to pass from a time from DEUS simulation in a time in second at this redshift
unit_t = 0.477522127117482E+17
Omega_matter_0 = 0.230000004172325  # ratio of matter in the Universe at z = 0,
# ratio of dark energy in the univers at z ~ 0
Omega_lambda_0 = 1 - Omega_matter_0
# 1 pc in meters, from the PDG, fixed value, no uncertainty
pc_to_meter = 3.08567758149 * 10**16
h = 0.72
L_box = (unit_l * 10**(-2))/((pc_to_meter/h) * 10**6)
redshift = 648/L_box - 1  # redshift  roughly 1 in DEUS , z ~ 2
print('z =', redshift)
# mass of one simu particle in Msun/h
mass_one_particle = 2.0217432046123 * 10**9
H0 = 100  # Hubble constant in h km s**-1 Mpc**-1
# newton constant in m**3 kg**-1 s**-2, from the PDG of 2016, the last 2 numbers are uncertain (31)
G_Newton = 6.67408 * 10**(-11)
G_Newton = G_Newton * 10**(-6)  # Newton constant in m kg**-1 (km/s)**2
meter_to_pc = 1/pc_to_meter
# 1 Msun in kg from the PDG, last number uncertain (9)
Msun_to_kg = 1.98848 * 10**30
kg_to_Msun = 1/Msun_to_kg
# G_Newton in pc Msun**-1 (km/s)**2
G_Newton = G_Newton * meter_to_pc/kg_to_Msun
G_Newton = G_Newton * 10**(-6)  # G_Newton in Mpc Msun**-1 (km/s)**2
# G_Newton in (Mpc/h) * (mass_one_particle)**(-1) * (km/s)**2
G_Newton = G_Newton * mass_one_particle
hsml = 0.005  # hsml in Mpc/h, (hydro smoothing length ?)
# formula of the Hubble parameter as a function of z
Ez = np.sqrt((Omega_lambda_0 + Omega_matter_0 * (1 + redshift)**3))
# formula of the Hubble expansion rate as a function of z in unit of (km/s)/(Mpc/h)
Hz = H0 * Ez
# ratio of matter in the Universe as a function of redshift
Omega_matter_z = Omega_matter_0 * ((1 + redshift)**3) * ((1/Ez)**2)
# critical density at redshift z in (mass_one_particle) / (Mpc/h)**3
rho_crit = 3*Hz*Hz/(8*np.pi*G_Newton)
rho_crit_0 = 3*H0*H0/(8*np.pi*G_Newton)
delta_vir_Brian_Norman = (18 * (np.pi)**2) + (82 * (Omega_matter_z - 1)) - (
    39 * (Omega_matter_z - 1)**2)  # delta vir of Brian and Norman 1997
print('Delta_vir_BN =', delta_vir_Brian_Norman)
print('18 * (np.pi)**2 =', 18 * (np.pi)**2)
print('rho_crit =', rho_crit)
# rho vir of Brian and Norman, in (mass_one_particle) / (Mpc/h)**3
rho_vir_Brian_Norman = delta_vir_Brian_Norman * rho_crit
print('rho_vir_BN =', rho_vir_Brian_Norman)
v_ren = unit_l/(unit_t * 10**5)  # put the velocity in km/s at this redshift
