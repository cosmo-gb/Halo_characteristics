#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:28:30 2022

@author: guillaume
"""

import numpy as np
import matplotlib.pyplot as plt



def profile_log_r_bin_hist(radius,r_min=0.01,r_max=1,N_bin=30,factor_random_mass=1):
    # It computes the density profile with an equal logarithmic shell size
    # It imposes shells of a given size, computed with r_min r_max and N_bin
    # Then it computes the number of particles in each shell
    # the density is simply N_part/V_shell
    # radius is the particles radii, not need to be sorted
    # radius, r_min and r_max should be in the same length unit
    # N_bin is the number of shells used for the density profile
    # rho_loc should be sorted as radius and in rho_crit unit (optional, to compute the scatter of the profile)
    # factor_random: take into account the mass differnce between a particle mass in the halo 
    # and a particle in the simulation DEUS, it can be =1 if I am using data from DEUS
    # r_bin_log: contains N_bin elements, radius_mean of each shell
    # rho_bin_log: contains N_bin elements, density profile in the shells
    # N_part_in_shell: contains N_bin elements, number of particles in each shell
    if r_min <= 0 :
        print('r_min should be strictly positive')
    r_log_bin = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) 
    N_part_in_shell, r_shell = np.histogram(radius, bins=r_log_bin) 
    N_part_not_res = len(np.where(radius < r_min)[0])
    dn, dr = np.zeros((N_bin+1),dtype=int), np.zeros((N_bin+1))
    dn[0] = N_part_not_res
    dn[1:] = N_part_in_shell
    dr[0] = r_log_bin[0]
    dr[1:] = r_log_bin[1:] - r_log_bin[:-1]
    r_log_bin = (r_log_bin[1:] + r_log_bin[:-1])/2
    Volume_shell = (4*np.pi/3)*(r_shell[1:]**3 - r_shell[:-1]**3)
    rho_log_bin = N_part_in_shell*factor_random_mass/Volume_shell
    return(r_log_bin,rho_log_bin,dn,dr) 

def NFW_func(x) :
    y = np.log(1+x) - x/(1+x)
    return(y)

def compute_mass_NFW(mass,r_s) :
    # it computes the mass inside r_s of an NFW profile
    Rvir = 1
    conc = Rvir/r_s
    M_in_sphere_r_s = mass * (np.log(2) - 0.5)/NFW_func(conc)
    return(M_in_sphere_r_s)

def compute_c_NFW_acc(r_data,res=0.01,N_cut=10) :
    # compute the concentration with accumulation mass method
    # following: https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
    # it assumes NFW
    # I cut in radius, and I recut and I recut ... until I found less than 2 particles
    mass = len(r_data)
    Rvir = 1
    r_cut = np.linspace(res,Rvir,2)
    t_keep = 1
    ind_stay = np.where( (r_cut[t_keep-1] < r_data) & (r_data < r_cut[t_keep]))[0]
    N_part_stay = len(ind_stay)
    while N_part_stay > 1 : # I stop when I found 1 particle or less in the radius bin considered
        # I define the radius bin r_cut
        r_cut = np.linspace(r_cut[t_keep-1],r_cut[t_keep],N_cut)
        for t in range(N_cut) :
            ind_t = np.where(r_data < r_cut[t])[0]
            M_t_data = len(ind_t) # number of particles inside the sphere of radius r_cut[t]
            # predicted number of particles inside the sphere of radius r_cut[t] for an NFW profile with r_s=r[t]
            M_t_th = compute_mass_NFW(mass,r_cut[t]) 
            # as long as M_data < M_NFW, it means that r=r_s is not reached
            if M_t_data > M_t_th :
                t_keep = t # I keep the iteration number for the next loop for define the next radius bins
                ind_stay = np.where( (r_cut[t_keep-1] < r_data) & (r_data < r_cut[t_keep]))[0]
                N_part_stay = len(ind_stay)
                break
    r_s = (r_cut[t_keep-1] + r_cut[t_keep])/2
    conc = Rvir/r_s
    return(conc)


def compute_c_NFW_acc_2(r_data) :
    # compute the concentration with accumulation mass method
    # following: https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
    # it assumes NFW
    N_part_tot = len(r_data)
    N_end = np.int(0.9 * N_part_tot)
    M_in_sphere_data = range(1,N_end+1)
    M_in_sphere_r_s_NFW = compute_mass_NFW(N_part_tot,r_data[0:N_end])
    diff = M_in_sphere_data - M_in_sphere_r_s_NFW
    out = np.diff(np.sign(diff))
    ind_sign_change = np.where(out != 0)[0]
    if len(ind_sign_change) == 1 :
        r_s = (r_data[ind_sign_change[0]] + r_data[ind_sign_change[0] - 1])/2
    else :
        print('problem')
        print(ind_sign_change)
        r_s = np.sum(r_data[ind_sign_change])/len(ind_sign_change)
    Rvir = 1
    conc = Rvir/r_s
    print(conc)
    plt.plot(r_data[0:N_end],M_in_sphere_data)
    plt.plot(r_data[0:N_end],M_in_sphere_r_s_NFW)
    plt.show()
    return(conc)
        
        
    

if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/cosmo_DE_z_impact/test/SA_halos/'
    N_samples = 3
    N_part = 1000
    res = 0.1
    N_bin = 20
    err_log, err_Child = np.zeros((N_samples)), np.zeros((N_samples))
    err_Einasto, err_Einasto_cov = np.zeros((N_samples)), np.zeros((N_samples))
    err_abg, err_abg_cov = np.zeros((N_samples)), np.zeros((N_samples))
    err_log_cov, err_Child_cov = np.zeros((N_samples)), np.zeros((N_samples))
    for s in range(N_samples) :
        pos = np.loadtxt(path+'halo_Npart_'+str(N_part)+'_abg_131_c_5_abc_111_'+str(s)+'.dat')
        conc_true = 5
        r_data = np.sqrt(pos[:,0]**2 + (pos[:,1])**2 + (pos[:,2])**2)
        conc = compute_c_NFW_acc(r_data)
        print(conc)
        compute_c_NFW_acc_2(np.sort(r_data))
        #r_log_bin, rho_log_bin, dn, dr = profile_log_r_bin_hist(r_data,r_min=res,N_bin=N_bin)