#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 06:40:10 2022

@author: guillaume
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# see e.g. https://terpconnect.umd.edu/~toh/spectrum/Smoothing.html

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


def triangular_smoothing(y) :
    y_new = (y[2:] + 2*y[1:-1] + y[0:-2])/4
    return(y_new)

def triangular_smoothing_5_points(y) :
    y_new = (y[4:] + 2*y[3:-1] + 3*y[2:-2] + 2*y[1:-3] + y[:-4])/9
    return(y_new)

def triangular_smoothing_7_points(y) :
    y_new = (y[6:] + 2*y[5:-1] + 3*y[4:-2] + 4*y[3:-3] + 3*y[2:-4] + 2*y[1:-5] + y[:-6])/16
    return(y_new)
    
def get_c_from_smooth_data(dn_dr_smooth,add_ind=0) :
    dn_dr_max = np.max(dn_dr_smooth)
    ind_max = np.where(dn_dr_smooth == dn_dr_max)[0]
    r_minus_2 = r_log_bin[ind_max+add_ind]
    r_minus_2_up = r_log_bin[ind_max+add_ind+1]
    r_minus_2_down = r_log_bin[ind_max+add_ind-1]
    conc = Rvir/r_minus_2
    conc_low = Rvir/r_minus_2_up
    conc_high = Rvir/r_minus_2_down
    dc = np.max([conc-conc_low,conc_high-conc])/2
    return(conc,dc)

def do_plot(c_t_3,dc_t_3,c_t_5,dc_t_5,c_SG_3,dc_SG_3,c_SG_5,dc_SG_5,N_samples) :
    plt.rc('font', family='serif',size=16)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    fig, ax = plt.subplots(figsize=[10,10]) # plot
    ax.tick_params(which='both', width=0.5, direction="in",
                   labeltop=False,labelbottom=True, labelleft=True, labelright=False)
    x = range(1,N_samples+1)
    y = np.ones((N_samples))
    
    plt.scatter(x,c_SG_3,c='g')
    plt.plot(x,np.mean(c_SG_3) * y,ls='-',c='g',label='% error mean, SG3')
    plt.plot(x,np.std(c_SG_3) * y,ls='-.',c='g',label='% error std, SG3')
    plt.hlines(np.mean(dc_SG_3),1,N_samples,ls='--',colors='g',label='bin width err')
    
    plt.scatter(x,c_SG_5,c='k',marker='.')
    plt.plot(x,np.mean(c_SG_5) * y,ls='-',c='k',label='% error mean, SG5')
    plt.plot(x,np.std(c_SG_5) * y,ls='-.',c='k')
    plt.hlines(np.mean(dc_SG_5),1,N_samples,ls='--',colors='k')
    
    plt.scatter(x,c_t_3,c='r')
    plt.plot(x,np.mean(c_t_3) * y,ls='-',c='r',label='% error mean, t3')
    plt.plot(x,np.std(c_t_3) * y,ls='-.',c='r')
    plt.hlines(np.mean(dc_t_3),1,N_samples,ls='--',colors='r')
    
    plt.scatter(x,c_t_5,c='b',)
    plt.plot(x,np.mean(c_t_5) * y,ls='-',c='b',label='% error mean, t5')
    plt.plot(x,np.std(c_t_5) * y,ls='-.',c='b')
    plt.hlines(np.mean(dc_t_5),1,N_samples,ls='--',colors='b')
    
    plt.ylabel('$\dfrac{c_{peak} - c_{true}}{c_{true}}$',labelpad=-5)
    plt.legend(loc='best', ncol=2, frameon=False)
    plt.title('$c_{peak}$ error on $N_{vir} = 10^'+str(np.int(np.log10(N_part)))+'$ particles Monte-Carlo haloes')
    path_and_name = path+'c_peak_error_on_Monte_Carlo_halos_with_'+str(N_part)+'_particles.pdf'
    #plt.savefig(path_and_name,format='pdf',transparent=True)
    plt.show()
    return()


if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/cosmo_DE_z_impact/test/SA_halos/'
    N_samples = 3
    c_t_3, dc_t_3 = np.zeros((N_samples)), np.zeros((N_samples))
    c_t_5, dc_t_5 = np.zeros((N_samples)), np.zeros((N_samples))
    c_t_7, dc_t_7 = np.zeros((N_samples)), np.zeros((N_samples))
    c_SG_3, dc_SG_3 = np.zeros((N_samples)), np.zeros((N_samples))
    c_SG_5, dc_SG_5 = np.zeros((N_samples)), np.zeros((N_samples))
    c_SG_7, dc_SG_7 = np.zeros((N_samples)), np.zeros((N_samples))
    N_part = 10000
    res = 0.05
    Rvir = 1
    N_bin = 20
    c_true = 5
    for s in range(N_samples) :
        pos = np.loadtxt(path+'halo_Npart_'+str(N_part)+'_abg_131_c_'+str(c_true)+'_abc_111_'+str(s)+'.dat')
        r_data = np.sqrt(pos[:,0]**2 + (pos[:,1])**2 + (pos[:,2])**2)
        r_log_bin,rho_log_bin,dn,dr = profile_log_r_bin_hist(r_data,r_min=res,N_bin=N_bin)
        print(dn)
        if s == 0 :
            print(r_log_bin)
        ############################################################################
        # 3 points triangular smoothing
        dn_dr_smooth = triangular_smoothing(rho_log_bin*r_log_bin**2)
        c_t_3[s], dc_t_3[s] = get_c_from_smooth_data(dn_dr_smooth,add_ind=1)
        #print('3 points c =',c_t_3[s],'err =',dc_t_3[s])
        ###############################################################################
        # 5 points triangular smoothing
        dn_dr_smooth = triangular_smoothing_5_points(rho_log_bin*r_log_bin**2)
        c_t_5[s], dc_t_5[s] = get_c_from_smooth_data(dn_dr_smooth,add_ind=2)
        #print('5 points c =',c_t_5[s],'err =',dc_t_5[s])
        ###############################################################################
        # 7 points triangular smoothing
        dn_dr_smooth = triangular_smoothing_7_points(rho_log_bin*r_log_bin**2)
        c_t_7[s], dc_t_7[s] = get_c_from_smooth_data(dn_dr_smooth,add_ind=3)
        #print('7 points c =',c_t_7[s],'err =',dc_t_7[s])
        ################################################################################
        # Savitzky-Golay algorithm for smoothing
        wl = 3
        pol = 2
        dn_dr_smooth = savgol_filter(rho_log_bin*r_log_bin**2,window_length=wl,polyorder=pol)
        c_SG_3[s], dc_SG_3[s] = get_c_from_smooth_data(dn_dr_smooth,add_ind=0)
        #print('Savitzky-Golay, wl=3 and pol=2, c =',c_SG_3[s],'err =',dc_SG_3[s])
        ##############################################################################
        # Savitzky-Golay algorithm for smoothing
        wl = 5
        pol = 3
        dn_dr_smooth = savgol_filter(rho_log_bin*r_log_bin**2,window_length=wl,polyorder=pol)
        c_SG_5[s], dc_SG_5[s] = get_c_from_smooth_data(dn_dr_smooth,add_ind=0)
        #print('Savitzky-Golay, wl=5 and pol=3, c =',c_SG_5[s],'err =',dc_SG_5[s])
        # Savitzky-Golay algorithm for smoothing
        wl = 7
        pol = 3
        dn_dr_smooth = savgol_filter(rho_log_bin*r_log_bin**2,window_length=wl,polyorder=pol)
        c_SG_7[s], dc_SG_7[s] = get_c_from_smooth_data(dn_dr_smooth,add_ind=0)
        #print('Savitzky-Golay, wl=7 and pol=3, c =',c_SG_7[s],'err =',dc_SG_7[s])
        
    ###########################################################################
    err_c_t_3 = (c_t_3 - c_true)/c_true
    err_dc_t_3 = dc_t_3/c_true
    err_c_t_5 = (c_t_5 - c_true)/c_true
    err_dc_t_5 = dc_t_5/c_true
    err_c_t_7 = (c_t_7 - c_true)/c_true
    err_dc_t_7 = dc_t_7/c_true
    err_c_SG_3 = (c_SG_3 - c_true)/c_true
    err_dc_SG_3 = dc_SG_3/c_true
    err_c_SG_5 = (c_SG_5 - c_true)/c_true
    err_dc_SG_5 = dc_SG_5/c_true
    err_c_SG_7 = (c_SG_7 - c_true)/c_true
    err_dc_SG_7 = dc_SG_7/c_true
    do_plot(err_c_t_3,err_dc_t_3,err_c_t_5,err_dc_t_5,
            err_c_SG_3,err_dc_SG_3,err_c_SG_5,err_dc_SG_5,
            N_samples)
    
    print('t3: ',np.mean(err_c_t_3),np.std(err_c_t_3),np.mean(err_dc_t_3))
    print('t5: ',np.mean(err_c_t_5),np.std(err_c_t_5),np.mean(err_dc_t_5))
    print('t7: ',np.mean(err_c_t_7),np.std(err_c_t_7),np.mean(err_dc_t_7))
    print('SG3: ',np.mean(err_c_SG_3),np.std(err_c_SG_3),np.mean(err_dc_SG_3))
    print('SG5: ',np.mean(err_c_SG_5),np.std(err_c_SG_5),np.mean(err_dc_SG_5))
    print('SG7: ',np.mean(err_c_SG_7),np.std(err_c_SG_7),np.mean(err_dc_SG_7))
        
        
        
        
        
        