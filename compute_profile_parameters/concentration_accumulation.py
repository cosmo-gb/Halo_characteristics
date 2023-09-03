#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 02 15:54:00 2023

@author: guillaume
"""


"""
This script contains 1 class Concentration_accumulation
which allows to compute the concentration of a halo 
following the mass accumulation technique described in
https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
see equation 6
It contains the method:
    - NFW_function
    - mass_in_r_s_NFW
    - compute_c_NFW_acc

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.signal import savgol_filter
import pandas as pd


from semi_analytical_halos.generate_smooth_halo import Smooth_halo
from semi_analytical_halos.density_profile import Profile




class Concentration_accumulation():

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

    def smooth(self, y: np.ndarray[float], box_pts: int,) -> np.ndarray[float]:
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def compute_c_NFW_acc(self, r_data: np.ndarray[float], box_pts: int=30, 
                          do_plot: bool=False,) -> Tuple[float, bool]:
        """
        This function computes the concentration of a halo with the accumulated mass technique.
        It follows the equation 6 of https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
        This technique assumes that the halo has a NFW profile.
        #############################################################################
        Input parameters:
        - r_data: np.array of float, radius of particles in the halo
        - box_pts: int, default = 30, for smoothing for convolution
        - do_plot: bool, default is False, plot or not the masses M(<r)
        #############################################################################
        Returns:
        - concentration: float, the halo concentration
        - ok: bool, if True there is only 1 unique solution, False otherwise
        """
        r_data_sorted = np.sort(r_data) # I need to sort the radii
        N_part_tot = len(r_data_sorted) # total number of particles
        # won't consider the very outer part of the halo,
        # as this part can be problematic for the change of sign
        N_end = int(0.9 * N_part_tot) # part of the halo
        M_in_sphere_data = range(1, N_end+1) # cumulative mass
        # then I compute the mass within r_s, assuming that r=r_s for all r
        M_in_sphere_r_s_NFW = self.mass_in_r_s_NFW(N_part_tot, r_data_sorted[0:N_end])
        M_in_sphere_r_s_NFW = self.smooth(M_in_sphere_r_s_NFW, box_pts)
        # difference between the masses: M(<r) - M_NFW(<r_s=r)
        diff = M_in_sphere_data - M_in_sphere_r_s_NFW 
        # the mass difference changes of sign at r=r_s
        # thus we look for this change of sign
        ind_sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(ind_sign_change) == 1: # case with only 1 solution
            ok = True
            #r_s = (r_data_sorted[ind_sign_change[0]] + r_data_sorted[ind_sign_change[0] - 1])/2
            #r_s = r_data_sorted[ind_sign_change[0]]
            #r_s = r_data_sorted[ind_sign_change[0]+1]
            r_s = r_data_sorted[ind_sign_change[0]-1]
            # I do not know why but it works better for N=1000
            #r_s = r_data_sorted[ind_sign_change[0]-7] 
        else: # case with at least 2 solutions
            ok = False
            r_s = np.sum(r_data_sorted[ind_sign_change])/len(ind_sign_change)
        conc = 1/r_s
        if do_plot:
            print(conc)
            plt.plot(r_data_sorted[0:N_end],
                    M_in_sphere_data, c="orange")
            plt.plot(r_data_sorted[0:N_end],
                    M_in_sphere_r_s_NFW, c="b")
            plt.show()
        return conc, ok
         
    def get_c_from_smooth_data(self, dn_dr_smooth: np.ndarray[float], r_log_bin: np.ndarray[float],
                               add_ind=0,) -> np.ndarray[float]:
        dn_dr_max = np.max(dn_dr_smooth)
        ind_max = np.where(dn_dr_smooth == dn_dr_max)[0]
        r_minus_2 = r_log_bin[ind_max+add_ind]
        r_minus_2_up = r_log_bin[ind_max+add_ind+1]
        r_minus_2_down = r_log_bin[ind_max+add_ind-1]
        conc = 1/r_minus_2
        conc_low = 1/r_minus_2_up
        conc_high = 1/r_minus_2_down
        dc = (conc_high - conc_low)/2
        #dc = np.max([conc-conc_low,conc_high-conc])/2
        return np.array([conc, dc]).T

if __name__ == '__main__':
    halo = Smooth_halo()
    kind_profile = {"kind of profile": "abg",
                    "concentration": 10,
                    "alpha": 1, "beta": 3, "gamma": 1}
    #my_halo = halo.smooth_halo_creation() #kind_profile, b_ax=0.5, c_ax=0.5)
    #data = my_halo["data"]
    #print("N_tot =",my_halo["N_tot"])
    #halo.plot_data(data[:,0],data[:,1])
    #halo.beauty_plot_colorbar(data)

    #r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    c_comp = Concentration_accumulation()
    profile = Profile()
    #conc, ok = c_acc.compute_c_NFW_acc(r_data)
    #print(conc)
    N_times = 30
    box_arr = [1, 2, 3, 4, 5]
    wl_arr = [2, 3, 4, 5]
    pol_arr = [1, 2, 3, 4, 5]
    c_acc, c_peak = np.zeros((N_times)), np.zeros((N_times,2*len(box_arr)))
    c_peak_sg = np.zeros((N_times, 2*len(pol_arr)*len(wl_arr)))
    for t in range(N_times):
        my_halo = halo.smooth_halo_creation(N_part=10000) #kind_profile, b_ax=0.5, c_ax=0.5)
        data = my_halo["data"]
        r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
        c_acc[t], ok = c_comp.compute_c_NFW_acc(r_data)
        if ok == False:
            print(ok, c_acc[t])
        out = profile.profile_log_r_bin_hist(r_data, N_bin=30)
        r_log_bin, rho_log_bin, N_part_in_shell = out
        rho_r2 = rho_log_bin * r_log_bin**2
        for b, box in enumerate(box_arr):
            rho_r2_smooth = c_comp.smooth(rho_r2, box)
            c_peak[t:t+1,b*2:(b+1)*2] = c_comp.get_c_from_smooth_data(rho_r2_smooth, r_log_bin)
        for w, wl in enumerate(wl_arr):
            for p, pol in enumerate(pol_arr):
                if pol < wl:
                    rho_r2_smooth = savgol_filter(rho_r2, window_length=wl, polyorder=pol)
                    beg = w*2*len(pol_arr)+p*2
                    end = beg + 2
                    #print(w, p, beg, end)
                    c_peak_sg[t:t+1, beg:end] = c_comp.get_c_from_smooth_data(rho_r2_smooth, r_log_bin)
                else :
                    break
    col = []
    for b in range(len(box_arr)):
        col += ["c_box_"+str(box_arr[b]),"dc_box_"+str(box_arr[b])]
    c_peak = pd.DataFrame(c_peak, columns=col)
    print(c_peak.mean(), c_peak.std())
    
    col = []
    for w in range(len(wl_arr)):
        for p in range(len(pol_arr)):
            col += ["c_wl_"+str(wl_arr[w])+"_"+str(pol_arr[p]),
                    "dc_wl_"+str(wl_arr[w])+"_"+str(pol_arr[p])]
    c_peak_sg = pd.DataFrame(c_peak_sg, columns=col)
    print(c_peak_sg.mean(), c_peak_sg.std())
    
