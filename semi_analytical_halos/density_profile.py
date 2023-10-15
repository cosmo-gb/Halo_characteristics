#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:21:51 2022

@author: guillaume
"""


'''
This script contains the class where I define everything usefull for the density profiles
'''

import numpy as np
import sys
import scipy.integrate as integrate
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from typing import Dict

import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()

class Profile:
    
    def profile_NFW(self, radius, conc, mass,):
        # profile NFW: rho(r) = rho_s/(x * (1 + x)**2), x=radius/r_s
        # radius should be in Rvir unit, rho_s should be in rho_crit, concentration, is dimensionless
        #Rvir = 1 # should be in the same dimension than the data, so I should let Rvir=1 and put the data (r_data) in Rvir unit
        r_s = 1/conc[0]
        A_c = np.log(1 + conc[0]) - conc[0]/(1 + conc[0]) # NFW factor
        rho_s = (mass/(4*np.pi*(r_s**3)*A_c)) # in rho_crit
        x_r = radius/r_s # dimensionless radius
        rho = rho_s * ( x_r**(-1) ) * (1 + x_r)**(-2) # in rho_crit
        return rho
    
    def abg_profile(self, r_minus_2, alpha, beta, gamma,):
        if alpha <=0 : # problematic case
            sys.exit('be carefull, alpha is negative or zero, it should be >0, alpha =',alpha)
        elif beta > 2 and gamma < 2 and gamma >= 0 : # standard case
            a = 1
            #print('good case: alpha =',alpha,' beta =',beta,' gamma =',gamma)
        elif gamma == 2 and beta > 2 : # gamma = 2 case, with beta > 2
            print('gamma = 2 and beta > 2')
            r_minus_2 = 0
        elif beta == 2 and gamma < 2 and gamma >= 0 : # beta = 2 case with 0 < gamma < 2
            print('beta = 2 and 0 <= gamma < 2')
            r_minus_2 = np.inf
        elif beta == 2 and gamma == 2 : # beta = 2 case with 0 < gamma < 2
            sys.exit('be carefull, beta = 2 and gamma = 2, so we have an isothermal-like profile')
            r_minus_2 = 'ill defined'
            r_s = 'ill defined'
        else : # either beta < 2 or gamma < 0 or gamma > 2
            sys.exit('be carefull, beta <= 2 or gamma < 0 or gamma > 2, beta =',beta,' gamma =',gamma)
            r_minus_2 = -1
        r_s = r_minus_2 * ((beta - 2)/(2 - gamma))**(1/alpha) # scale radius in Rvir unit
        #print('r_s =',r_s)
        n_profile = lambda x: (x**(-gamma)) * (1 + x**alpha)**((gamma - beta)/alpha) 
        return(n_profile, r_s)
    
    def Einasto_profile(self, alpha_Einasto=0.17,):
        if alpha_Einasto <= 0 :
            print('alpha_Einasto should be positive but =',alpha_Einasto)
        n_profile = lambda x: np.exp( (-2/alpha_Einasto) * (np.power(x,alpha_Einasto) - 1) )
        return(n_profile)

    def single_slope_profile(self, slope,):
        n_profile = lambda x: x**(-slope)
        return(n_profile)

    def compute_log_slope_abg(self, x_edges, alpha=1, beta=3, gamma=1,):
        N_x_strict_positive = len(np.where(x_edges[1:] > 0)[0])
        N_x = len(x_edges)
        x_bin = (x_edges[1:] + x_edges[:-1])/2
        if x_edges[0] == 0 and N_x_strict_positive == N_x - 1  : # case with unresolved area, where I put log_slope=0
            log_slope = -gamma + (gamma-beta)/(1 + np.power(x_bin[1:],-alpha))
            log_slope = np.append(0,log_slope)
        elif x_edges[0] > 0 and N_x_strict_positive == N_x - 1  : # case without unresolved area
            log_slope = -gamma + (gamma-beta)/(1 + np.power(x_bin,-alpha))
        else :
            print('test ????')
            sys.exit('x should not be negative or have multiples 0 values')
        return(log_slope)
    
    def dlog_M_dlog_r_NFW(self, x,):
        num = (x**2)
        den = ((1+x)**2) * (np.log(1+x) -x/(1+x))
        y = num/den
        return(y)
    
    def compute_log_slope_Einasto(self, x_edges, alpha_Einasto=0.17,):
        # analytical formula of the logarithmic derivative of the Einasto profile:
        # dlog(rho)/dlog(r) = -2 * (r/r_s)**(alpha_Einasto)
        N_x_strict_positive = len(np.where(x_edges[1:] > 0)[0])
        N_x = len(x_edges)
        x_bin = (x_edges[1:] + x_edges[:-1])/2
        if x_edges[0] == 0 and N_x_strict_positive == N_x - 1  : # case with unresolved area, where I put log_slope=0
            log_slope = -2 * np.power(x_bin[1:],alpha_Einasto) 
            log_slope = np.append(0,log_slope)
        elif x_edges[0] > 0 and N_x_strict_positive == N_x - 1  : # case without unresolved area
            log_slope = -2 * np.power(x_bin,alpha_Einasto) 
        else :
            sys.exit('x should not be negative or have multiples 0 values')
        return(log_slope)
    
    def compute_mass_Einsato(self, r, r_minus_2, alpha, rho_minus_2,):
        # see Equation 16 of Springel+08: mass included inside the radius r for an Einsto profile with alpha parameter
        # M(<r) = ((4 pi r_minus_2**3 rho_minus_2)/alpha) * exp((3*ln(alpha) + 2 -ln(8))/alpha) * gamma(a,x)
        # with gamma(a,x) the lower incomplete gamma function
        # a=3/alpha, x=(2/alpha) * (r/r_minus_2)**alpha
        a = 3/alpha
        x = (2/alpha) * np.power(r/r_minus_2,alpha)
        gam = gamma(a) * gammainc(a,x) # gamma(a,x) the lower incomplete gamma function
        fac_cst = ( ( (4 * np.pi * r_minus_2**3) * rho_minus_2)/alpha) * np.exp( (3 * np.log(alpha) + 2 - np.log(8))/alpha)
        mass = fac_cst * gam
        dM_dr = 4 * np.pi * rho_minus_2 * (r**2) * np.exp((-2/alpha)*((r/r_minus_2)**alpha - 1))
        mass_slope = dM_dr * r/mass # logarithmic derivative of the mass with respect the radius
        return(mass,mass_slope)
        
    def compute_mass_NFW(self, r, r_s, rho_s,):
        # see Equation 3.72 in the thesis of Stref Martin
        x = r/r_s
        M_in_r = (4 * np.pi * rho_s * r_s**3) * (np.log(1+x) - x/(x+1))
        return(M_in_r)
    
    def NFW_function(self, x: np.ndarray[float],) -> np.ndarray[float]:
        """
        This computes the NFW function y = ln(1+x) - x/(1+x)
        #############################################################################
        Input parameters:
        - x: radius of the halo in scale radius r_s unit
        #############################################################################
        Returns:
        - outputs of the NFW function
        """
        return np.log(1+x) - x/(1+x)
    
    def mass_in_r_s_NFW(self, Mass_total: float, r_s: np.ndarray[float],) -> np.ndarray[float]:
        """
        This function computes the mass inside r_s of an NFW profile.
        It is using the equation 6 of 
        https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
        #############################################################################
        Input parameters:
        - Mass_total: total mass in the halo (e.g. Mass_total=Mvir if the virial radius is considered)
        it can be a number of particles as it is a mass in unit of particle mass
        - r_s: scale radius: r_s = 1/concentration in unit of the halo size
        r_s should be given in unit of the halo size
        #############################################################################
        Returns:
        - mass contained within r_s = M(<r_s)
        """
        return Mass_total * (np.log(2) - 0.5)/self.NFW_function(1/r_s)
    
    def mass_NFW(self, radius, conc, mass, r_min,):
        # masses of a halo of mass mass and concentration conc, inside spheres of radius radius, according to NFW
        # see equation 2 of Bhattacharya+13: https://iopscience.iop.org/article/10.1088/0004-637X/766/1/32/pdf
        # radius should be in Rvir unit
        assert radius[0] > r_min, "the first element of radius should be larger than the resolution r_min"
        radius = np.insert(radius, 0, r_min)
        M_in_sphere_r = self.NFW_function(conc * radius) * mass/self.NFW_function(conc)
        #M_in_sphere_rmin = self.NFW_func(conc * r_min) * mass/self.NFW_func(conc)
        M_shell_r = np.zeros((len(radius)-1))
        M_shell_r = M_in_sphere_r[1:] - M_in_sphere_r[:-1]
        #M_shell_r[0] = M_in_sphere_r[0] - M_in_sphere_rmin
        return M_in_sphere_r, M_shell_r
    
    def get_rho_s(self, n_x, r_min=0, r_max=1,):
        #r_s, n_x, log_slope = self.deal_with_kind_profile(kind_profile, r_bin, R_max)
        n_x_2 = lambda x: (x**(2)) * n_x(x)
        ####################################################################### set everything in r_s unit:
        #R_min, R_max, res = R_min/r_s, R_max/r_s, res/r_s
        #r_bin = r_bin/r_s
        ####################################################################### total number of particles in the halo and normalisation of the density profile
        my_int_tot = integrate.quad(lambda x: n_x_2(x), r_min, r_max) # volume in r_s**3 unit
        n_s = 1/(4*np.pi*my_int_tot[0]) # scale number density in r_s**(-3) unit
        return(n_s)
    
    def get_rho_minus_2(self, concentration, N_part, kind_profile, R_min=0, R_max=1,):
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
    
    def deal_with_kind_profile(self, kind_profile: Dict, r_bin: float=-1, R_max: float=1,):
        # deal with kind_profile:
        if kind_profile["kind of profile"] == "abg": # case of an alpha beta, gamma density profile
            concentration = kind_profile["concentration"]
            r_minus_2 = R_max/concentration # r_s = r_minus_2 for an NFW profile only
            alpha = kind_profile["alpha"]
            beta = kind_profile["beta"]
            gamma = kind_profile["gamma"]
            n_profile, r_s = self.abg_profile(r_minus_2, alpha, beta, gamma)
            if type(r_bin) == np.ndarray:
                log_slope = self.compute_log_slope_abg(r_bin/r_s, alpha, beta, gamma)
        elif kind_profile["kind of profile"] == "Einasto" : # case of an Einasto density profile
            concentration = kind_profile["concentration"]
            r_s = R_max/concentration # r_s = r_minus_2 for an Einasto profile
            alpha_Einasto = kind_profile["alpha"]
            n_profile = self.Einasto_profile(alpha_Einasto=alpha_Einasto)
            if type(r_bin) == np.ndarray:
                log_slope = self.compute_log_slope_Einasto(r_bin/r_s,alpha_Einasto) 
        elif kind_profile["kind of profile"] == "single slope" : # case of a single slope profile
            delta = kind_profile["delta"]
            n_profile = self.single_slope_profile(delta)
            r_s = R_max
            if type(r_bin) == np.ndarray:
                log_slope = delta * np.ones((len(r_bin)))
        else: 
            print("I did not implemented this kind of profile yet")
        if type(r_bin) == np.ndarray:
            return r_s, n_profile, log_slope
        else :
            return r_s, n_profile
                
    def compute_logarithmic_derivative(self, r, rho,):
        # this function computes the logarithmic derivative of the function rho(r)
        # using (rho(r+h)-rho(r-h))/(2*h)
        log_r = np.log10(r)
        log_rho = np.log10(rho)
        num = log_rho[2:] - log_rho[0:-2]
        den = log_r[2:] - log_r[0:-2]
        alpha = num/den
        alpha_first = (log_rho[1]-log_rho[0])/(log_r[1]-log_r[0])
        alpha_last = (log_rho[-1]-log_rho[-2])/(log_r[-1]-log_r[-2])
        alpha = np.append(alpha_first,alpha)
        alpha = np.append(alpha,alpha_last)
        return(alpha)
    
    def profile_log_r_bin_hist_old_old(self, radius, r_min=0.01, r_max=1, N_bin=30, factor_random_mass=1,):
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
        r_log_bin = (r_log_bin[1:] + r_log_bin[:-1])/2
        Volume_shell = (4*np.pi/3)*(r_shell[1:]**3 -r_shell[:-1]**3)
        rho_log_bin = N_part_in_shell*factor_random_mass/Volume_shell
        return r_log_bin, rho_log_bin, N_part_in_shell
    
    def profile_log_r_bin_hist_old(self, radius, r_min=0.01, r_max=1, N_bin=30,
                               factor_random_mass=1, dn_dr=False):
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
        #if r_min <= 0:
        #    print('r_min should be strictly positive')
        assert r_min > 0, "r_min should be strictly positive"
        r_log_bin = np.logspace(np.log10(r_min), np.log10(r_max), N_bin+1) 
        N_part_in_shell, r_shell = np.histogram(radius, bins=r_log_bin)
        Volume_shell = (4*np.pi/3)*(r_shell[1:]**3 - r_shell[:-1]**3)
        rho_log_bin = N_part_in_shell * factor_random_mass/Volume_shell
        if dn_dr:
            N_part_not_res = len(np.where(radius < r_min)[0])
            dn, dr = np.zeros((N_bin+1), dtype=int), np.zeros((N_bin+1))
            dn[0] = N_part_not_res
            dn[1:] = N_part_in_shell
            dr[0] = r_log_bin[0]
            dr[1:] = r_log_bin[1:] - r_log_bin[:-1]
            r_log_bin = (r_log_bin[1:] + r_log_bin[:-1])/2
            return r_log_bin, rho_log_bin, N_part_in_shell, dn, dr
        else:
            r_log_bin = (r_log_bin[1:] + r_log_bin[:-1])/2
            return r_log_bin, rho_log_bin, N_part_in_shell
        
    def profile_log_r_bin_hist(self, radius, r_min=0.01, r_max=1, 
                                   N_bin=30, factor_random_mass=1,) -> Dict[str,np.ndarray]:
        assert r_min > 0, "r_min should be strictly positive"
        r_shell = np.logspace(np.log10(r_min), np.log10(r_max), N_bin+1)
        N_part_in_shell, r_shell = np.histogram(radius, bins=r_shell)
        size_shell = r_shell[1:] - r_shell[:-1]
        volume_shell = (4*np.pi/3)*(r_shell[1:]**3 - r_shell[:-1]**3)
        rho = N_part_in_shell * factor_random_mass/volume_shell
        N_part_not_res = np.array([len(np.where(radius < r_min)[0])])
        radius = (r_shell[1:] + r_shell[:-1])/2
        dic = {"radius": radius, # average of the radius of the bin edge i.e. of the shell edge
               "rho": rho, # density in the bin = density(r_bin[i]<r<r_bin[i+1])
               "r_shell": r_shell, # radius of the edge of the bins i.e. of the shell
               "N_part_in_shell": N_part_in_shell, # number of particles in each shell
               "size_shell": size_shell, # radial size of each shell
               "N_part_not_res": N_part_not_res, # number of particles at r<r_min
               }
        return dic
        
    def plot_profile(self, r_bin, rho_bin,):
        plt.rc('font', family='serif')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots() # plot
        ax.tick_params(which='both', width=0.5, direction="in",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        ax.loglog(r_bin,rho_bin,c='r') 
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.set_xlabel('$r$ $[R_{vir}]$')
        ax.set_ylabel('$n(r)$ $[R_{vir}^{-3}]$',labelpad=-2)
        plt.show()
        return()
    
    def plot_two_profiles(self, r_bin, rho_bin_1, rho_bin_2,):
        plt.rc('font', family='serif')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots() # plot
        ax.tick_params(which='both', width=0.5, direction="in",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        ax.loglog(r_bin,rho_bin_1,c='r') 
        ax.loglog(r_bin,rho_bin_2,c='b') 
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.set_xlabel('$r$ $[R_{vir}]$')
        ax.set_ylabel('$n(r)$ $[R_{vir}^{-3}]$',labelpad=-2)
        plt.show()
        return()
    
    def plot_profile_r_2(self, r_bin, rho_bin_r_2,):
        plt.rc('font', family='serif')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots() # plot
        ax.tick_params(which='both', width=0.5, direction="in",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        ax.loglog(r_bin,rho_bin_r_2,c='r') 
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.set_xlabel('$r$ $[R_{vir}]$')
        ax.set_ylabel(r'$n(r)$ $\times$ $r^2$ $[R_{vir}^{-1}]$',labelpad=-2)
        plt.show()
        return()
    
    
if __name__ == '__main__':
    from semi_analytical_halos.generate_smooth_halo import Smooth_halo
    halo = Smooth_halo()
    my_profile = Profile()
    N_part = 1000
    N_bin = 10
    N_halos = 100
    for h in range(N_halos):
        my_halo = halo.smooth_halo_creation(N_part=N_part, N_bin=N_bin) #kind_profile, b_ax=0.5, c_ax=0.5)
        data = my_halo["data"]
        r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
        dic = my_profile.profile_log_r_bin_hist(r_data, N_bin=N_bin)
        print(h)
        print(dic)
        