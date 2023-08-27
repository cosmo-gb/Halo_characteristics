#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:43:59 2022

@author: guillaume
"""

'''
This script contains 1 class Halo_with_sub which allows to generate
semi-analytical halo with subhalos from analytical density profile.
The class Halo_with_sub contains the following methods:
    - plot_data_two
    - subhalo_mass_function
    - get_subhalo_masses
    - select_subhalos
    - c_M
    - get_concentration_from_mass
    - get_rho_minus_2
    - generate_halo_with_sub: this is the main method of this class
    - generate_many

'''

# import standard libraries
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import random
from typing import Dict, Callable

# import my class
from generate_smooth_halo import Smooth_halo # for generate smooth halo
from compute_tidal_radii import Tidal_radius # for compute the tidal radius of subhalo

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
    
    def subhalo_mass_function(self, M_sub: float, m_min: float, m_max: float, 
                              delta: float) -> Callable[[float], float] :
        """
        Generate the subhalo mass function dN/dm.
        dN/dm is the number of subhalo of in the mass range [m, m+dm]:
        dN/dm = A * m**(-delta)
        Input parameters:
        - M_sub: total subhalo mass
        - m_min, m_max: subhalo mass minimum and maximum allowed
        - delta: parameter of the previous formula
        Returns:
        - lambda function: dN_dm(m) i.e. number of subhalo of mass range [m, m+dm]
        """
        if delta != 2 :
            A = M_sub * (-delta + 2)/(m_max**(-delta+2) - m_min**(-delta+2))
        else :
            A = M_sub/np.log(m_max/m_min)
        dN_dm = lambda m: A * m**(-delta)        
        return dN_dm
    
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
    
    def generate_halo_with_sub(self, kind_profile_main: Dict={}, 
                               kind_profile_sub: Dict={}, 
                               kind_profile_pos_sub: Dict={},
                               M_tot: float=100, m_part: float=0.001,
                               R_min_main: float=0, R_max_main: float=1,
                               f_sub: float=0.1, m_min: float=0.01, m_max: float=10,
                               delta: float=2, N_bin_main: int=30, 
                               a_ax_main: float=1, b_ax_main: float=1, c_ax_main: float=1,
                               N_sub_m_bin: int=30, fac_fake: float=5,
                               res: float=0.001, r_shell_binning: str="logarithmic",
                               verbose: bool=False,) -> Dict:
        ###################### set default kind of profiles ##########################################
        if kind_profile_main == {}: # default density profile of the main halo (NFW with c=10)
            kind_profile_main = {"kind of profile": "abg",
                                 "concentration": 10,
                                 "alpha": 1, "beta": 3, "gamma": 1}
        if kind_profile_sub == {}: # default density profile of the subhalos (NFW with c=10)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # the other analytical density profiles are not implemented yet,
            # see tidal radius r_t_Jacobi_smooth_Springel funstion
            kind_profile_sub = {"kind of profile": "abg",
                                "alpha": 1, "beta": 3, "gamma": 1}
        if kind_profile_pos_sub == {}: # default spatial distributions of the subhalos (NFW with c=1)
            kind_profile_pos_sub = {"kind of profile": "abg",
                                    "concentration": 1,
                                    "alpha": 1, "beta": 3, "gamma": 1}
        ######################## set the halo mass ####################################################
        # say that the mass is in unit of Milky-Way like halo, 
        # so Mass_main = 100 MW = 10**14 Msun/h and m_part = 0.001 MW = 10**9 M_sun/h by default    
        # res, R_min_main and R_max_main are in unit of the main halo size (so R_max_main = 1)
        M_main = M_tot * (1 - f_sub) # mass of the main smooth halo in MW mass
        N_main = int(M_main/m_part) # total number of particles in the main smooth halo
        M_sub = f_sub * M_tot # total mass of the subhalos in MW mass
        ######################## main smooth halo generation ##########################################
        main_halo = self.smooth_halo_creation(kind_profile_main, N_part=N_main, N_bin=N_bin_main,
                                              R_min=R_min_main, R_max=R_max_main, res=res,
                                              r_shell_binning=r_shell_binning,
                                              a_ax=a_ax_main, b_ax=b_ax_main, c_ax=c_ax_main,)
        # main_halo contains: data, N_tot, N_part_bin, r_bin, r, r_ell, r_s, n_s, n_x
        # r_bin, r, r_ell, r_s are in unit of R_max_main
        # n_s is in unit r_s**(-3), n_x is the dimensionless profile n(x) of the main halo
        ######################## subhalo mass function (fake) #########################################
        # I start by generating more (fake) suhbalos than necessary and I will remove the overage later
        # I am doing this because I have to remove subhalo mass because of tidal forces
        # fake subhalo mass function
        dN_dm = self.subhalo_mass_function(M_sub*fac_fake, m_min, m_max, delta) 
        N_sub_tot = integrate.quad(lambda m: dN_dm(m), m_min, m_max) # total number of subhalos (proxy)
        N_sub_tot = self.around_integer_from_float(N_sub_tot[0]) # total number of subhalos (proxy)
        if verbose:
            print("Estimate of number of fake subhalos generated = ", N_sub_tot)
        ######################## masses of subhalos ##################################################
        N_sub_bin, m_sub, N_sub_tot, m_bin = self.get_subhalo_masses(dN_dm, m_min, m_max, N_sub_m_bin)
        ######################## concentrations of subhalos ##########################################
        # from m_sub I generate c_sub, with normal distributions (see Batthacharya+13)
        c_sub = self.get_concentration_from_mass(m_sub) 
        r_minus_2_sub = 1/c_sub # in unit of the size of each subhalo
        ######################## get rho_minus_2 from m_sub and c_sub ################################
        N_part_in_sub = self.around_integer_from_float(m_sub/m_part)# Number of particles inside subhalos
        R_max_sub = R_max_main * (m_sub/M_main)**(1/3) # radius of each sub_halo in the unit of the main halo
        # then I select only particles in a sphere of r_t for each subhalo
        ######################### set positions of the subhalos ######################################
        # number of bins for subhalo positions
        N_bin_subalo_pos = int(np.round(30 * np.log10(1 + N_sub_tot/100), decimals=0)) 
        if N_bin_subalo_pos < 10 :
            print("The number of bin for the subhalos distribution of positions is :", N_bin_subalo_pos)
            print("This is small thus I reset it to 10")
            N_bin_subalo_pos = 10
        sub_pos = self.smooth_halo_creation(kind_profile_pos_sub, N_part=N_sub_tot, 
                                            N_bin=N_bin_subalo_pos, R_min=res,
                                            a_ax=a_ax_main, b_ax=b_ax_main, c_ax=c_ax_main)
        sub_pos = sub_pos["data"] # positions of the subhalos
        r_sub = np.sqrt(sub_pos[:,0]**2 + sub_pos[:,1]**2 + sub_pos[:,2]**2) # in the halo unit
        N_sub_fake = len(sub_pos) # number of subhalo
        ########################## subhalos generation ##############################################
        # I start to generate more subhalos than needed
        # initialisation: halo_tot will contain the positions of all particles in the total halo at the end
        halo_tot = main_halo["data"] 
        # N_part_sub_fin will contain the number of particles contained in each halo after tidal radius
        N_part_sub_fin = np.zeros((N_sub_fake), dtype=int) 
        # !!! hard coded: r_bin_sub is only used for the computation of the mass
        r_min, r_max, N_bin = 0.0001, 1, 1000
        r_bin_sub = np.logspace(np.log10(r_min), np.log10(r_max), N_bin+1) # in unit of the subhalo size
        r_t_sub = np.zeros((N_sub_fake)) # will contain tidal radius of all subhalos
        if verbose:
            print("Number of fake subhalos generated = ", N_sub_fake)
        N_min_in_sub = m_min/m_part # minimal number of particles in a subhalo (before tidal effect)
        if N_min_in_sub < 10:
            print("Be carefull, you are generating subhalos with ",N_min_in_sub," particles")
        # number of bins for particles inside subhalos, in principle different for each subhalo
        N_bin_subalos = np.array(np.round(10 * np.log10(1 + N_part_in_sub/N_min_in_sub),
                                          decimals=0), dtype=int)
        ########################### loop on each individual subhalo ###############################
        for s in range(N_sub_fake) : # loop on each individual subhalo
            kind_profile_sub["concentration"] = c_sub[s]
            ####################### generation of the subhalo s ###################################
            # be carefull, I assumed that my subhalos are all sphericals
            sub_halo = self.smooth_halo_creation(kind_profile_sub, N_part=N_part_in_sub[s], 
                                                 N_bin=N_bin_subalos[s],
                                                 R_min=0, R_max=R_max_sub[s], res=res,
                                                 r_shell_binning=r_shell_binning)
                                                 #a_ax=a_ax_main,b_ax=b_ax_main,c_ax=c_ax_main)
            # parameter of the subhalo s
            r_s, n_x = self.deal_with_kind_profile(kind_profile_sub) 
            # tidal radius of the subhalo s
            r_t_sub[s] = self.r_t_Jacobi_smooth_Springel(kind_profile_main, r_sub[s], kind_profile_sub,
                                                         r_s, n_x, r_bin_sub)
            r_t_sub[s] *= R_max_sub[s] # set in the main halo unit
            ####################### apply tidal effect ############################################
            # now I apply tidal effect and remove the outer part of the subhalo s
            sub_halo = sub_halo["data"]
            r_in_sub = np.sqrt(sub_halo[:,0]**2 + sub_halo[:,1]**2 + sub_halo[:,2]**2) # in the halo unit
            if np.max(r_in_sub) < r_t_sub[s] : # normally this is not possible ! 
                # case where the tidal radius is larger than the subhalo size
                N_part_sub_fin[s] = 0
                sub_halo_pos = sub_halo
                print("Problem: the tidal radius should be smaller than the subhalo size")
            else : # otherwise: remove the particles outside the tidal radius
                ind_tide = np.where(r_in_sub < r_t_sub[s])[0]
                sub_halo_pos = sub_halo[ind_tide] # keep only the particles inside the tidal radius
                N_part_sub_fin[s] = len(ind_tide) # number of particles kept
            # Finally I add the particles of this subhalo s to the total halo
            halo_tot = np.append(halo_tot, sub_halo_pos+sub_pos[s], axis=0) 
        ##################### Remove the overage of subhalos ##########################################
        # fraction of subhalos that need to be removed: 
        N_main_get = main_halo["N_tot"] # total number of particles in the smooth main halo
        N_tot = M_tot/m_part # total number of particles that I want
        f_sub_fake = np.sum(N_part_sub_fin)/N_tot # subhalo mass fraction that I get with the overage
        f_remove = f_sub/f_sub_fake # inverse fraction of what I need to remove
        ##################### Generate the final (real i.e. without overage) subhalo mass function ####
        dN_dm = self.subhalo_mass_function(M_sub, m_min, m_max, delta) # real subhalo mass function
        # then I select a fraction of subhalo that I will keep
        ind_sub_use = self.select_subhalos(dN_dm, m_part, m_max, N_sub_m_bin,
                                           N_part_sub_fin*m_part, f_remove)
        N_part_sub_halos = N_part_sub_fin[ind_sub_use] # number of particles in the subhalo kept
        f_sub_end = np.sum(N_part_sub_halos)/N_tot # final real fraction of subhalo obtained
        print("Desired fraction of subhalos =",f_sub)
        print("Obtained fraction of subhalo at the end =",f_sub_end)
        ###################### Finally I add these subhalos to the main halo ###########################
        N_subhalos = len(ind_sub_use) # total number of subhalos kept
        halo_sub = halo_tot[N_main_get:] # contains all subhalo including fake ones
        halo_sub_keep = np.zeros((1,3))
        N_beg, N_end = 0, 0
        for s in range(N_sub_fake): # loop on all fake subhalo
            N_s_fake = N_part_sub_fin[s] # number of particles in the subhalo s after tidal effect
            ind_fake = np.where(ind_sub_use == s)[0]
            if len(ind_fake) == 1: # then I keep this subhalo s: it is a real one
                N_end = N_end +  N_s_fake
                halo_sub_keep = np.append(halo_sub_keep, halo_sub[N_beg:N_end], axis=0)
                N_beg = N_beg +  N_s_fake
            else : # then I do not keep this subhalo s, it is a fake subhalo
                N_end = N_end +  N_s_fake
                N_beg = N_beg +  N_s_fake
        # the final total halo (main smooth + real subhalos) is halo_tot
        halo_tot = np.append(main_halo["data"], halo_sub_keep[1:], axis=0)
        ######################## Verbose ###############################################################
        if verbose:
            f_sub_init = np.sum(N_part_in_sub)/N_tot
            print("Fraction of fake subhalos generated =", f_sub_init, 
                  " with fac_fake=", fac_fake, " before applying tidal radius")
            print("Fraction of subhalos generated =", f_sub_fake, 
                  " with fac_fake=", fac_fake, " after applying tidal radius")
            print("Total number of particles in the main halo =", N_main_get)
            print("Total number of particles in the subhalos =", np.sum(N_part_sub_halos))
            print("Total number of particles in the halo =", len(halo_tot))
        halo_with_sub = {"halo_tot": halo_tot, "subhalo_pos": sub_pos, "N_sub_tot": N_sub_tot,
                         "m_sub": m_sub, "R_max_sub": R_max_sub, "N_part_in_sub": N_part_in_sub,
                         "N_part_sub_fin": N_part_sub_fin, "main_halo": main_halo, 
                         "f_sub_end": f_sub_end, "N_subhalos": N_subhalos}
        return halo_with_sub
    
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
    halo = Halo_with_sub() ############################ Be carefull, here !!!!!!!!!
    c_main, c_sub, c_sub_pos = 10, 10, 1
    kind_profile_main = {"kind of profile": "abg",
                         "concentration": 10,
                         "alpha": 1, "beta": 3, "gamma": 1}
    kind_profile_sub = {"kind of profile": "abg",
                        "alpha": 1, "beta": 3, "gamma": 1}
    kind_profile_pos_sub = {"kind of profile": "abg",
                            "concentration": 1,
                            "alpha": 1, "beta": 3, "gamma": 1}
    my_halo = halo.generate_halo_with_sub() #kind_profile_main, kind_profile_sub, kind_profile_pos_sub)
    data = my_halo["halo_tot"]
    main_halo = my_halo["main_halo"]
    N_part_main = main_halo["N_tot"]
    #halo.plot_data(data[:,0],data[:,1])
    halo.beauty_plot_colorbar(data)
    #halo.plot_data_two(data[:,0], data[:,1], N_part_main)
    #halo.generate_many(10)

