#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:43:59 2022

@author: guillaume
"""

'''
This script is the new version of a code to generate halo from profile with subhalos
'''

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import random

from generate_smooth_halo import Smooth_halo
from compute_tidal_radii import Tidal_radius

class Halo_with_sub(Smooth_halo, Tidal_radius):
    
    def plot_data_two(self,x,y,N,x_min=-1,x_max=1,y_min=-1,y_max=1) :
        plt.rc('font', family='serif')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots() # plot
        ax.tick_params(which='both', width=0.5, direction="in",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        #####################################################################################""
        ax.scatter(x[0:N],y[0:N],c='r',s=0.01)
        ax.scatter(x[N:],y[N:],c='b',s=0.01)
        ########################################################################################
        #ax.set_xlim(np.min(x),np.max())
        #ax.set_ylim(y_min,y_max)
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        return()
    
    def subhalo_mass_function(self,M_sub,m_min,m_max,delta) :
        # dN/dm = A * m**(-delta)
        if delta != 2 :
            A = M_sub * (-delta + 2)/(m_max**(-delta+2) - m_min**(-delta+2))
        else :
            A = M_sub/np.log(m_max/m_min)
        dN_dm = lambda m: A * m**(-delta)        
        return(dN_dm)
    
    def get_subhalo_masses(self,dN_dm,m_min,m_max,N_sub_m_bin) :
        m_bin = 10**(np.linspace(np.log10(m_min), np.log10(m_max), N_sub_m_bin+1))
        N_sub_bin = np.zeros((N_sub_m_bin+1),dtype=int) # it contains N_sub_m_bin + 1 elements
        for b in range(N_sub_m_bin) :
            # need to change that also !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            my_int = integrate.quad(lambda m: dN_dm(m), m_bin[b], m_bin[b+1])
            N_sub_bin[b] = int(np.round(my_int[0],decimals=0)) # number of subhalos by subhalo mass bin
        N_sub_tot = np.sum(N_sub_bin)
        m_sub = np.zeros((N_sub_tot))
        N_start, N_end = 0, N_sub_bin[0]
        for b in range(N_sub_m_bin) :
            m_sub[N_start:N_end] = np.array(np.random.uniform(m_bin[b], m_bin[b+1],N_sub_bin[b]),dtype=np.float32)
            N_start += N_sub_bin[b]
            N_end += N_sub_bin[b+1]
        return(N_sub_bin,m_sub,N_sub_tot,m_bin)
    
    def select_subhalos(self, dN_dm, m_min, m_max, N_sub_m_bin, m_sub, f_remove) :
        ''' Select a fraction f_remove of suhalos that has been created and
        that are contained in m_sub
        '''
        m_bin = 10**(np.linspace(np.log10(m_min), np.log10(m_max), N_sub_m_bin+1))
        N_sub_bin = np.zeros((N_sub_m_bin+1),dtype=int) # it contains N_sub_m_bin + 1 elements
        ind_b_use = np.array((),dtype=int)
        for b in range(N_sub_m_bin) :
            my_int = integrate.quad(lambda m: dN_dm(m), m_bin[b], m_bin[b+1])
            # number of subhalos by subhalo mass bin
            N_sub_bin[b] = self.around_integer_from_float(my_int[0]) 
            ind_bin_b = np.where( (m_sub <= m_bin[b+1]) & (m_sub > m_bin[b]) )[0]
            N_sel = len(ind_bin_b) * f_remove
            N_sel_int = self.around_integer_from_float(N_sel)
            if N_sel_int <= len(ind_bin_b) :
                # then I select the subhalo that I keep
                ind_sel = random.sample(range(len(ind_bin_b)),N_sel_int)
            else :
                ind_sel = range(len(ind_bin_b))
            ind_b_use = np.append(ind_b_use,ind_bin_b[ind_sel])
        return(ind_b_use)
                   
    def c_M(self,mass,A=5,B=-0.1) :
        # mass should be in unit 10**12 M_sun/h
        mass = mass/100
        # mass should be in unit 10**14 M_sun/h
        # see the thesis of Matthieu Schaller
        c = A * np.power(mass,B)
        return(c)
    
    def get_concentration_from_mass(self,m_sub,c_pb=0.1):
        # c(M) (see Batthacharya+13, normal distribution)
        # mass should be in unit 10**14 M_sun/h
        # see the thesis of Matthieu Schaller
        c_mean_sub = self.c_M(m_sub)
        sigma_c_sub = 0.3 * c_mean_sub
        c_sub = np.random.normal(loc=c_mean_sub,scale=sigma_c_sub)
        ind_pb = np.where(c_sub <= c_pb)[0] 
        i = 0
        while len(ind_pb) > 0:
            print('I generated a negative concentration so I restart')
            c_mean_sub = self.c_M(m_sub)
            sigma_c_sub = 0.3 * c_mean_sub
            c_sub = np.random.normal(loc=c_mean_sub,scale=sigma_c_sub)
            ind_pb = np.where(c_sub <= c_pb)[0] 
            if i > 10 :
                print('WTFUCK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break
        return(c_sub)
    
    def get_rho_minus_2(self,concentration,N_part,kind_profile,R_min=0,R_max=1) :
        r_minus_2 = 1/concentration # r_minus_2 in unit of the subhalo size r_sub_max
        #r_bin = np.logspace(np.log10(R_min),np.log10(R_max),N_bin+1) # no influence on rho_minus_2 I think
        #r_s, n_x, log_slope = self.deal_with_kind_profile(kind_profile, r_bin)
        n_x = self.Einasto_profile(alpha_Einasto=kind_profile[2])
        n_x_2 = lambda x: (x**(2)) * n_x(x)
        ####################################################################### total number of particles in the halo and normalisation of the density profile
        if isinstance(concentration, (np.ndarray)) : 
            N_sub = len(concentration)
            rho_minus_2 = np.zeros((N_sub))
            for s in range(N_sub) :
                my_int_tot = integrate.quad(lambda x: n_x_2(x), R_min/r_minus_2[s], R_max/r_minus_2[s]) # volume in r_minus_2**3 unit
                rho_minus_2[s] = (N_part[s]/(r_minus_2[s]**3))/(4*np.pi*my_int_tot[0]) # scale number density
        else :
            my_int_tot = integrate.quad(lambda x: n_x_2(x), R_min/r_minus_2, R_max/r_minus_2) # volume in r_minus_2**3 unit
            rho_minus_2 = (N_part/(r_minus_2**3))/(4*np.pi*my_int_tot[0]) # scale number density, in unit of the size of the subhalo cube
        return(rho_minus_2)
    
    def generate_halo_with_sub(self, kind_profile_main, kind_profile_sub, kind_profile_pos_sub,
                               M_tot=100,m_part=0.001,R_min_main=0,R_max_main=1,
                               f_sub=0.1,m_min=0.01,m_max=10,delta=2,
                               N_bin_main=30,a_ax_main=1,b_ax_main=1,c_ax_main=1,
                               N_sub_m_bin=30,fac_fake=5,
                               res=0.001,r_shell_binning='logarithmic',verbose=False):
        ''' This function is the main function of this class.
        '''
        # say that the mass is in unit of Milky-Way like halo, 
        # so Mass_main = 100 MW = 10**14 Msun/h and m_part = 0.001 MW = 10**9 M_sun/h by default    
        # res, R_min_main and R_max_main are in unit of the main halo size (so R_max_main = 1)
        M_main = M_tot * (1 - f_sub) # mass of the main smooth halo in MW mass
        N_main = int(M_main/m_part) # total number of particles in the main smooth halo
        M_sub = f_sub * M_tot # total mass of the subhalos in MW mass
        ####################################################################### main smooth halo generation
        main_halo = self.smooth_halo_creation(kind_profile_main,N_part=N_main,N_bin=N_bin_main,
                                              R_min=R_min_main,R_max=R_max_main,res=res,
                                              r_shell_binning=r_shell_binning,
                                              a_ax=a_ax_main,b_ax=b_ax_main,c_ax=c_ax_main)
        # main_halo contains: data, N_tot, N_part_bin, r_bin, r, r_ell, r_s, n_s, n_x
        # r_bin, r, r_ell, r_s are in unit of R_max_main
        # n_s is in unit r_s**(-3), n_x is the dimensionless profile n(x) of the main halo
        ####################################################################### subhalo mass function
        # I start by generating more (fake) suhbalos than necessary and I will remove the overage later
        # I am doing this because I have to remove subhalo mass because of tidal forces
        dN_dm = self.subhalo_mass_function(M_sub*fac_fake, m_min, m_max, delta) # fake subhalo mass function
        N_sub_tot = integrate.quad(lambda m: dN_dm(m), m_min, m_max) # approximation of the total number of subhalos
        N_sub_tot = self.around_integer_from_float(N_sub_tot[0]) # approximation of the total number of subhalos
        if verbose:
            print('Number of fake subhalos generated =',N_sub_tot)
        ####################################################################### masses of the subhalos
        N_sub_bin, m_sub, N_sub_tot, m_bin = self.get_subhalo_masses(dN_dm,m_min,m_max,N_sub_m_bin)
        ####################################################################### concentrations of subhalos
        # from m_sub I generate c_sub, with normal distributions (see Batthacharya+13)
        c_sub = self.get_concentration_from_mass(m_sub) 
        r_minus_2_sub = 1/c_sub # in unit of the size of each subhalo
        ####################################################################### get rho_minus_2 from m_sub and c_sub
        N_part_in_sub = m_sub/m_part # Number of particles inside each subhalos
        N_part_in_sub = self.around_integer_from_float(N_part_in_sub) 
        R_max_sub = R_max_main * (m_sub/M_main)**(1/3) # radius of each sub_halo in the unit of the main halo
        # then I select only particles in a sphere of r_t for each subhalo
        ####################################################################### positions of the subhalos
        sub_pos = self.smooth_halo_creation(kind_profile_pos_sub,N_part=N_sub_tot,N_bin=100,R_min=res,
                                            a_ax=a_ax_main,b_ax=b_ax_main,c_ax=c_ax_main)
        sub_pos = sub_pos[0] # positions of the subhalos
        r_sub = np.sqrt(sub_pos[:,0]**2 + sub_pos[:,1]**2 + sub_pos[:,2]**2) # in the halo unit
        N_sub_fake = len(sub_pos)
        ####################################################################### subhalos generation
        # I start to generate a lot of subhalo
        halo_tot = main_halo[0] # initialisation: will contain the positions of all particles in the total halo
        N_part_sub_fin = np.zeros((N_sub_fake), dtype=int) # R_max and N_part of each subhalo
        r_min, r_max, N_bin = 0.001, 1, 1000
        r_test = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) # in unit of the subhalo size
        r_t_sub = np.zeros((N_sub_fake))
        if verbose:
            print('Number of fake subhalos generated =',N_sub_fake)
        for s in range(N_sub_fake) : # loop on each individual subhalo
            ################################################################### subhalo s generation
            kind_profile_use = kind_profile_sub
            kind_profile_use[1] = c_sub[s]
            # be carefull, I assumed that my subhalos are all spherical
            sub_halo = self.smooth_halo_creation(kind_profile_use,N_part=N_part_in_sub[s],N_bin=10,
                                                 R_min=0,R_max=R_max_sub[s],res=res,
                                                 r_shell_binning=r_shell_binning)
                                                 #a_ax=a_ax_main,b_ax=b_ax_main,c_ax=c_ax_main)
            ##################################################################################################
            r_s, n_x, log_slope = self.deal_with_kind_profile(kind_profile_use, r_test, r_max) 
            r_t_sub[s] = self.r_t_Jacobi_smooth_Springel(kind_profile_main, r_sub[s], kind_profile_use, r_s, n_x, r_test)
            r_t_sub[s] *= R_max_sub[s] # in the main halo unit
            ####################################################################################################
            # now I apply tidal effect and remove the outer part of subhalos
            r_in_sub = np.sqrt(sub_halo[0][:,0]**2 + sub_halo[0][:,1]**2 + sub_halo[0][:,2]**2) # in the halo unit
            if np.max(r_in_sub) < r_t_sub[s] : # normally this is not possible ! the tidal radius is smaller than the subhalo radius
                N_part_sub_fin[s] = 0
                sub_halo_pos = sub_halo[0]
                print('Problem: the tidal radius should be smaller than the subhalo radius')
            else :
                ind_tide = np.where(r_in_sub < r_t_sub[s])[0]
                sub_halo_pos = sub_halo[0][ind_tide]
                N_part_sub_fin[s] = len(ind_tide)
            ############################################################################################"
            # I add the particles of this subhalo s to the total halo
            halo_tot = np.append(halo_tot,sub_halo_pos+sub_pos[s],axis=0) 
        ##################################################################################################
        # Now I finally remove the overage of subhalos that I previously generated
        # fraction of subhalos that need to be removed: 
        N_main_get = main_halo[1] # total number of particles in my smooth main halo
        N_tot = M_tot/m_part
        f_sub_fake = np.sum(N_part_sub_fin)/N_tot # subhalo mass fraction that I get with the overage
        f_remove = f_sub/f_sub_fake # inverse fraction of what I need to remove
        ####################################################################### subhalo mass function
        dN_dm = self.subhalo_mass_function(M_sub, m_min, m_max, delta) # real subhalo mass function
        # then I select a fraction of subhalo that I will keep
        ind_sub_use = self.select_subhalos(dN_dm, m_part, m_max, N_sub_m_bin,
                                           N_part_sub_fin*m_part, f_remove)
        N_part_sub_halos = N_part_sub_fin[ind_sub_use]
        f_sub_end = np.sum(N_part_sub_halos)/N_tot
        print('fraction of subhalo desired =',f_sub)
        print('fraction of subhalo obtained =',f_sub_end)
        ###############################################################################################
        # finally I add these subhalos to the main halo
        N_sub_halos = len(ind_sub_use)
        halo_sub = halo_tot[N_main_get:]
        halo_sub_keep = np.zeros((1,3))
        N_beg, N_end = 0, 0
        for s in range(N_sub_fake) :
            N_s_fake = N_part_sub_fin[s] 
            ind_fake = np.where(ind_sub_use == s)[0]
            if len(ind_fake) == 1 :
                N_end = N_end +  N_s_fake
                halo_sub_keep = np.append(halo_sub_keep,halo_sub[N_beg:N_end],axis=0)
                N_beg = N_beg +  N_s_fake
            else :
                N_end = N_end +  N_s_fake
                N_beg = N_beg +  N_s_fake
        halo_tot = np.append(main_halo[0],halo_sub_keep[1:],axis=0)
        ################################################################################################
        #print('f_sub_final =',f_sub_fake)
        f_sub_init = np.sum(N_part_in_sub)/N_tot
        #print('fraction of subhalos_sub_initial =',f_sub_init)
        if verbose:
            print('Total number of particles in the main halo =',N_main_get)
            print('Total number of particles in the subhalos =',np.sum(N_part_sub_halos))
            print('Total number of particles in the halo =',len(halo_tot))
        return(halo_tot,sub_pos,N_sub_tot,m_sub,R_max_sub,N_part_in_sub,N_part_sub_fin,main_halo,f_sub_end)
    
    def generate_many(self,N=2):
        c_main, c_sub, c_sub_pos = 10, 10, 1
        kind_profile_main = ['abg',c_main]
        kind_profile_sub = ['abg',c_sub]
        kind_profile_pos_sub = ['abg',c_sub_pos]
        f_sub = np.zeros((N))
        for n in range(N) :
            print('n =',n)
            data = self.generate_halo_with_sub(kind_profile_main,kind_profile_sub,kind_profile_pos_sub)
            f_sub[n] = data[-1]
        print('f_sub_mean =',np.mean(f_sub))
        print('f_sub_std =',np.std(f_sub))
        return()
    
    
if __name__ == '__main__':
    halo = halo_with_sub() ############################ Be carefull, here !!!!!!!!!
    c_main, c_sub, c_sub_pos = 10, 10, 1
    kind_profile_main = ['abg',c_main]
    kind_profile_sub = ['abg',c_sub]
    kind_profile_pos_sub = ['abg',c_sub_pos]
    my_halo = halo.generate_halo_with_sub(kind_profile_main, kind_profile_sub, kind_profile_pos_sub,
                                          a_ax_main=1,b_ax_main=0.2,c_ax_main=0.2)
    data = my_halo[0]
    main_halo = my_halo[-2]
    N_part_main = main_halo[1]
    #halo.plot_data(data[:,0],data[:,1])
    halo.beauty_plot_colorbar(data)
    #halo.plot_data_two(data[:,0], data[:,1], N_part_main)
    #halo.generate_many(10)

