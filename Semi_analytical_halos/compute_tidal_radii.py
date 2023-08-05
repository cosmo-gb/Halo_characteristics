#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:07:07 2022

@author: guillaume
"""

'''
This script computes the tidal radii of a subhalo inside a main halo using different methods
I need to check if there is an issue with rho_s and also maybe I can accelerate this script I think
I also check that I got similar results than the Figure 3.8 in the thesis of Stref Martin.
'''

import numpy as np
import matplotlib.pyplot as plt

from density_profile import profile

class tidal_radius(profile):
    
    #def __init__(self, kind_main, kind_subhalo):
    #    self.kind_main = kind_main
    #    self.kind_subhalo = kind_subhalo
        
    def r_t_Jacobi_1(self,R_main,kind_subhalo,r_s,n_x,r):
        # Ok in fact I think that all mass are in unit of the mass of the main halo: to be checked
        # M_main and M_sub should be in the same unit
        if kind_sub[0] == 'abg' :
            if len(kind_subhalo) == 2 : # NFW case
                rho_s = self.get_rho_s(n_x,0,r_max/r_s) # rho_s is in r_s**3 unit
                #r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1)
                m_in_r = self.compute_mass_NFW(r,r_s,rho_s/r_s**3) # in the same unit as M_sub
            else :
                print('not NFW')
        else :
            print('not abg')
        # Equation 3.99 in Stref thesis
        r_t_test = R_main * (m_in_r/3)**(1/3) # in unit of the main halo
        chi_2 = (r_t_test - r)**2
        ind_test_ok = np.where(chi_2 == np.min(chi_2))[0]
        if len(ind_test_ok) > 1:
            print('problem')
        else:
            r_t = r[ind_test_ok]
        return(r_t)
    
    def r_t_Jacobi_2(self,kind_main,R_main,kind_subhalo,r_s,n_x,r):
        # Ok in fact I think that all mass are in unit of the mass of the main halo: to be checked
        # M_main and M_sub should be in the same unit
        #r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
        r_s_main, n_x_main, log_slope_main = self.deal_with_kind_profile(kind_main, r, r_max) 
        rho_s_main = self.get_rho_s(n_x_main,0,1/r_s_main) # rho_s is in r_s**3 unit
        M_main_in_R_main = self.compute_mass_NFW(R_main,r_s_main,rho_s_main/r_s_main**3)
        if kind_sub[0] == 'abg' :
            if len(kind_subhalo) == 2 : # NFW case
                rho_s = self.get_rho_s(n_x,0,1/r_s) # rho_s is in r_s**3 unit
                #r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1)
                m_in_r = self.compute_mass_NFW(r,r_s,rho_s/r_s**3) # in the same unit as M_sub
            else :
                print('not NFW')
        else :
            print('not abg')
        # Equation 3.100 in Stref thesis
        r_t_test = R_main * (m_in_r/(3*M_main_in_R_main))**(1/3) # in unit of the main halo
        chi_2 = (r_t_test - r)**2
        ind_test_ok = np.where(chi_2 == np.min(chi_2))[0]
        if len(ind_test_ok) > 1:
            print('problem')
        else:
            r_t = r[ind_test_ok]
        return(r_t)
    
    def r_t_Jacobi_smooth_Stref(self,kind_main,R_main,kind_subhalo,r_s,n_x,r):
        # Ok in fact I think that all mass are in unit of the mass of the main halo: to be checked
        # M_main and M_sub should be in the same unit
        #r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
        r_s_main, n_x_main, log_slope_main = self.deal_with_kind_profile(kind_main, r, r_max) 
        rho_s_main = self.get_rho_s(n_x_main,0,1/r_s_main) # rho_s is in r_s**3 unit
        M_main_in_R_main = self.compute_mass_NFW(R_main,r_s_main,rho_s_main/r_s_main**3)
        if kind_sub[0] == 'abg' :
            if len(kind_subhalo) == 2 : # NFW case
                rho_s = self.get_rho_s(n_x,0,1/r_s) # rho_s is in r_s**3 unit
                m_in_r = self.compute_mass_NFW(r,r_s,rho_s/r_s**3) # in the same unit as M_sub
            else :
                print('not NFW')
        else :
            print('not abg')
        # Equation 3.105 in Stref thesis
        dlog_M_dlog_r = self.dlog_M_dlog_r_NFW(R_main/r_s_main)
        r_t_test = R_main * (m_in_r/(3*M_main_in_R_main*(1-dlog_M_dlog_r/3)))**(1/3) # in unit of the main halo
        chi_2 = (r_t_test - r)**2
        ind_test_ok = np.where(chi_2 == np.min(chi_2))[0]
        if len(ind_test_ok) > 1:
            print('problem')
        else:
            r_t = r[ind_test_ok]
        return(r_t)
    
    def r_t_Jacobi_smooth_Springel(self,kind_main,R_main,kind_sub,r_s,n_x,r,r_max=1):
        # Ok in fact I think that all mass are in unit of the mass of the main halo: to be checked
        # M_main and M_sub should be in the same unit
        #r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
        r_s_main, n_x_main, log_slope_main = self.deal_with_kind_profile(kind_main, r, r_max) 
        rho_s_main = self.get_rho_s(n_x_main,0,1/r_s_main) # rho_s is in r_s**3 unit
        M_main_in_R_main = self.compute_mass_NFW(R_main,r_s_main,rho_s_main/r_s_main**3)
        if kind_sub[0] == 'abg' :
            if len(kind_sub) == 2 : # NFW case
                rho_s = self.get_rho_s(n_x,0,1/r_s) # rho_s is in r_s**3 unit
                m_in_r = self.compute_mass_NFW(r,r_s,rho_s/r_s**3) # in the same unit as M_sub
            else :
                print('not NFW')
        else :
            print('not abg')
        # Equation 12 of Springel+08
        dlog_M_dlog_r = self.dlog_M_dlog_r_NFW(R_main/r_s_main)
        r_t_test = R_main * (m_in_r/(M_main_in_R_main*(2-dlog_M_dlog_r)))**(1/3) # in unit of the main halo
        chi_2 = (r_t_test - r)**2
        ind_test_ok = np.where(chi_2 == np.min(chi_2))[0]
        if len(ind_test_ok) > 1:
            print('problem')
        else:
            r_t = r[ind_test_ok]
        return(r_t)
    
    def r_t_dens(self,kind_main,R_main, r_s, n_x, r):
        # Ok in fact I think that all mass are in unit of the mass of the main halo: to be checked
        r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
        r_s_main, n_x_main, log_slope_main = self.deal_with_kind_profile(kind_main, r, r_max) 
        rho_s_main = self.get_rho_s(n_x_main,0,1/r_s_main) # rho_s is in r_s**3 unit
        rho_main = n_x_main(R_main/r_s_main) * rho_s_main/(r_s_main**3)
        ########################################################################
        # be carefull, r_s is in unit of the subhalo size here
        rho_s = self.get_rho_s(n_x, 0, 1/r_s) # rho_s is in r_s**3 unit
        rho_sub = n_x(r/r_s) * rho_s/(r_s**3)
        chi_2 = (rho_sub - rho_main)**2
        ind_test_ok = np.where(chi_2 == np.min(chi_2))[0]
        if len(ind_test_ok) > 1:
            print('problem')
        else:
            r_t = r[ind_test_ok] # in unit of the subhalo size here
        return(r_t)
    
    def r_t_of_R(self, kind_main, r, kind_subhalo, r_s, n_x):
        N_bin = len(r)
        r_t_Jac_1, r_t_Jac_2, r_t_dens = np.zeros((N_bin)), np.zeros((N_bin)), np.zeros((N_bin))
        r_t_Jac_smooth_Stref, r_t_Jac_smooth_Springel = np.zeros((N_bin)), np.zeros((N_bin))
        for n in range(N_bin):
            r_t_Jac_1[n] = self.r_t_Jacobi_1(r[n], kind_subhalo,r_s, n_x, r)
            r_t_Jac_2[n] = self.r_t_Jacobi_2(kind_main,r[n], kind_subhalo, r_s, n_x, r)
            r_t_Jac_smooth_Stref[n] = self.r_t_Jacobi_smooth_Stref(kind_main,r[n], kind_subhalo, r_s, n_x, r)
            r_t_Jac_smooth_Springel[n] = self.r_t_Jacobi_smooth_Springel(kind_main,r[n], kind_subhalo, r_s, n_x, r)
            r_t_dens[n] = self.r_t_dens(kind_main,r[n], r_s, n_x, r)
        return(r_t_Jac_1, r_t_Jac_2, r_t_Jac_smooth_Stref, r_t_Jac_smooth_Springel, r_t_dens)
    
    def plot_r_t(self,r,r_t,r_s,path_and_name=''):
        plt.rc('font', family='serif')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots() # plot
        ax.tick_params(which='both', width=0.5, direction="in",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        ax.loglog(r,r_t[0]/r_s,c='r',ls='--',label='1 Jac') 
        ax.loglog(r,r_t[1]/r_s,c='g',ls='--',label='2 Jac')
        ax.loglog(r,r_t[2]/r_s,c='b',ls='--',label='Stref')
        ax.loglog(r,r_t[3]/r_s,c='b',ls='-.',label='Springel')
        ax.loglog(r,r_t[4]/r_s,c='purple',ls='--',label='dens')
        ax.set_xlim(0.1,200)
        ax.set_ylim(10**(-2),10**2)
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.set_xlabel('$R$ $[kpc]$',labelpad=-3)
        ax.set_ylabel('$r_t/r_s$',labelpad=0)
        #ax.legend(loc='best',bbox_to_anchor=(0.53, 0.45, 0.5, 0.5),fontsize=20,frameon=False,ncol=1)
        ax.legend(loc='center',bbox_to_anchor=(0.6, 0.95),fontsize=20,frameon=False,ncol=2)
        if path_and_name != '' :
            plt.savefig(path_and_name)
        plt.show()
        return()
            
    
    
if __name__ == '__main__':
    c_main = 12
    kind_main = ['abg',c_main]
    c_sub = 60
    kind_sub = ['abg',c_sub]
    tide = tidal_radius()
    M_main, R_main = 10**6, 0.1
    M_sub = 10**2
    r_min = 0.0001
    N_bin = 10000
    r_min, r_max, N_bin = 0.0001, 1, 1000
    r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
    r_s, n_x, log_slope = tide.deal_with_kind_profile(kind_sub, r, r_max) 
    r = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
    r_t = tide.r_t_of_R(kind_main, r, kind_sub, r_s, n_x)
    #print(r_t)
    print(r)
    print(r_t[0])
    path = '../../../../../DEUSS_648Mpc_2048_particles/Results/keep/2022_10_02/'
    name = 'r_t_R.pdf'
    path_and_name = path + name
    tide.plot_r_t(r*200, r_t, r_s)   