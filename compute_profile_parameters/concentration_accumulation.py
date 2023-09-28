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

import sys
print(sys.path)
# computation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# typing
from typing import Tuple, List
# stat
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

# my code
#from semi_analytical_halos.generate_smooth_halo import generate_smooth_halo
#Smooth_halo = generate_smooth_halo.Smooth_halo
#from semi_analytical_halos.generate_smooth_halo import Smooth_halo
#from generate_smooth_halo import Smooth_halo
#from generate_smooth_halo import Smooth_halo
from semi_analytical_halos.generate_smooth_halo import Smooth_halo



class Concentration_accumulation(Smooth_halo):

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

    def build_list(self, wl_arr: List[int], pol_arr: List[int]) -> List[List[int]]:
        my_list = []
        for w in wl_arr:
            for p in pol_arr:
                if p < w:
                    my_list += [[w,p]]
        return my_list

    def compute_many_c_smoothing(self, r_data: np.ndarray, N_bin: int,
                                 box_arr: List[int], wl_pol: List[List[int]]) \
                                 -> Tuple[np.ndarray]:
        out = self.profile_log_r_bin_hist(r_data, N_bin=N_bin)
        r_log_bin, rho_log_bin = out["radius"], out["rho"]
        rho_r2 = rho_log_bin * r_log_bin**2
        c_peak, dc_peak = np.zeros((len(box_arr))), np.zeros((len(box_arr)))
        c_peak_sg, dc_peak_sg = np.zeros((len(wl_pol))), np.zeros((len(wl_pol)))
        for b, box in enumerate(box_arr):
            rho_r2_smooth = self.smooth(rho_r2, box)
            c = self.get_c_from_smooth_data(rho_r2_smooth, r_log_bin)
            c_peak[b:b+1], dc_peak[b:b+1] = c[0][0], c[0][1]
        for e, wp in enumerate(wl_pol):
            rho_r2_smooth = savgol_filter(rho_r2, window_length=wp[0], polyorder=wp[1])
            c = self.get_c_from_smooth_data(rho_r2_smooth, r_log_bin)
            c_peak_sg[e:e+1], dc_peak_sg[e:e+1] = c[0][0], c[0][1]
        return c_peak, dc_peak, c_peak_sg, dc_peak_sg

    def test_many_computation(self, box_arr: List[int], wl_arr: List[int], pol_arr: List[int],
                              N_halos: int, N_part: int, N_bin: int) -> Tuple[pd.DataFrame]:
        # initialisation
        wl_pol = self.build_list(wl_arr, pol_arr)
        c_acc = np.zeros((N_halos))
        c_peak, dc_peak = np.zeros((N_halos, len(box_arr))), np.zeros((N_halos, len(box_arr)))
        c_peak_sg, dc_peak_sg = np.zeros((N_halos, len(wl_pol))), np.zeros((N_halos, len(wl_pol)))
        # loop on many halos
        for t in range(N_halos):
            my_halo = self.smooth_halo_creation(N_part=N_part, N_bin=N_bin) #kind_profile, b_ax=0.5, c_ax=0.5)
            data = my_halo["data"]
            r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
            c_acc[t], ok = self.compute_c_NFW_acc(r_data)
            if ok == False:
                print(ok, c_acc[t])
            ##################################################################################
            cs = self.compute_many_c_smoothing(r_data, N_bin, box_arr, wl_pol)
            c_peak[t:t+1], dc_peak[t:t+1] = cs[0], cs[1]
            c_peak_sg[t:t+1], dc_peak_sg[t:t+1] = cs[2], cs[3]
            ##################################################################################
        col, dc_col = [], []
        for b in range(len(box_arr)):
            col += ["c_box_"+str(box_arr[b])]
            dc_col += ["dc_box_"+str(box_arr[b])]
        c_peak = pd.DataFrame(c_peak, columns=col)
        dc_peak = pd.DataFrame(dc_peak, columns=dc_col)
        ####################################################################
        col, dc_col = [], []
        for wp in wl_pol:
            col += ["c_wl_"+str(wp[0])+"_pol_"+str(wp[1])]
            dc_col += ["dc_wl_"+str(wp[0])+"_pol_"+str(wp[1])]
        c_peak_sg = pd.DataFrame(c_peak_sg, columns=col)
        dc_peak_sg = pd.DataFrame(dc_peak_sg, columns=dc_col)
        c_acc = pd.DataFrame(c_acc, columns=["c_acc"])
        return c_acc, c_peak, dc_peak, c_peak_sg, dc_peak_sg
    
    def do_plot(self, c_acc: pd.DataFrame, c_peak: pd.DataFrame, dc_peak: pd.DataFrame,
                c_peak_sg: pd.DataFrame, dc_peak_sg: pd.DataFrame) -> None:
        c_acc = c_acc.values
        fig, ax = plt.subplots(nrows=2, figsize=(10,7), sharey=True)
        ax[0].errorbar(range(len(c_peak_sg.mean())),
                    c_peak_sg.mean(), yerr=c_peak_sg.std().values, # + dc_peak_sg.mean().values,
                    fmt="o", label="c_peak_sg")
        ax[0].axhline(10, ls="-", color="r", label="true value")
        ax[0].axhline(np.mean(c_acc), ls="-", color="g", label="c_acc")
        ax[0].axhline(np.mean(c_acc)+np.std(c_acc), ls="--", color="g")
        ax[0].axhline(np.mean(c_acc)-np.std(c_acc), ls="--", color="g")
        ax[0].set_xticks(ticks=range(len(c_peak_sg.mean())),
                        labels=c_peak_sg.columns, rotation=45)
        ax[0].xaxis.tick_top()
        ax[0].set_yticks(ticks=range(8,13))
        ax[0].set_ylim(7,13)
        ax[0].grid(True)
        ax[0].set_ylabel("concentration")
        ax[0].legend()
        ##############################################################################""
        ax[1].errorbar(range(len(c_peak.mean())),
                    c_peak.mean(), yerr=c_peak.std().values, # + dc_peak_sg.mean().values,
                    fmt="o", label="c_peak_conv")
        ax[1].axhline(10, ls="--", color="r", label="true value")
        ax[1].axhline(np.mean(c_acc), ls="-", color="g", label="c_acc")
        ax[1].axhline(np.mean(c_acc)+np.std(c_acc), ls="--", color="g")
        ax[1].axhline(np.mean(c_acc)-np.std(c_acc), ls="--", color="g")
        ax[1].set_xticks(ticks=range(len(c_peak.mean())),
                labels=c_peak.columns, rotation=45)
        ax[1].set_yticks(ticks=range(8,13))
        ax[1].set_ylim(7,13)
        ax[1].grid(True)
        ax[1].set_ylabel("concentration")
        ax[1].legend()
        plt.show()

    def residual_log(self, param, x_data, y_data, mass,):
        y_th = self.profile_NFW(x_data, param, mass)
        # the squared of res is minimized if loss='linear' (default)
        return np.log10(y_data) - np.log10(y_th)
    
    def residual_Child18(self, param, x_data, dn, dr, mass,):
        # it computes the residual for the fit, see the equation 5 of Child+18
        # https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
        rho_NFW = self.profile_NFW(x_data, param, mass)
        dn_dr_NFW = 4 * np.pi * x_data**2 * rho_NFW
        # the squared of residual is minimized if the option loss='linear' (default) in least_squares
        return np.sqrt(((dn/dr - dn_dr_NFW)**2)/(dn/dr**2)) 
    
    def residual_Bhattacharya13(self, param, x_data, dn, mass, r_min,):
        # it computes the residual of the fit, see the equation 4 of Bhattacharya+13
        # https://iopscience.iop.org/article/10.1088/0004-637X/766/1/32/pdf
        M_in_sphere_r, M_shell_r = self.mass_NFW(x_data, param, mass, r_min)
        return ((dn - M_shell_r)**2)/dn

    def find_bining(self,r_data, N_bin,):
        for b in range(1,N_bin):
            # decrease the binning
            N_bin_use = N_bin - b
            out = self.profile_log_r_bin_hist(r_data, N_bin=N_bin_use)
            N_part_in_shell = out["N_part_in_shell"]
            if np.min(N_part_in_shell) > 0:
                return out, N_bin_use
            
    def fit_concentration(self, r_data, N_bin, methods, p0=np.array([100]), bounds=(0.1, 1000)):
        mass = len(r_data)
        out = self.profile_log_r_bin_hist(r_data, N_bin=N_bin)
        N_part_in_shell = out["N_part_in_shell"]
        if np.min(N_part_in_shell) == 0:
            # decrease the binning
            out, N_bin = self.find_bining(r_data, N_bin)
            print("N_bin_use = ",N_bin)
        r, rho = out["radius"], out["rho"]
        size_shell = out["size_shell"]
        N_part_in_shell = out["N_part_in_shell"]
        r_shell = out["r_shell"]
        dic = {}
        if "log_lm" in methods:
            # No bounds => Levenberg-Marquardt Algorithm: 
            # Implementation and Theory,” Numerical Analysis, ed. G. A. Watson, 
            # Lecture Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
            lsq = least_squares(self.residual_log, p0, method="lm", args=(r, rho, mass))
            dic["c_log_lm"] = lsq["x"]
        if "log_trf" in methods:
            # with bounds => Trust Region reflective
            # M. A. Branch, T. F. Coleman, and Y. Li, 
            # “A Subspace, Interior, and Conjugate Gradient Method 
            # for Large-Scale Bound-Constrained Minimization Problems,”
            # SIAM Journal on Scientific Computing, Vol. 21, Number 1, pp 1-23, 1999.
            lsq = least_squares(self.residual_log, p0, bounds=bounds,
                                method="trf", args=(r, rho, mass))
            dic["c_log_trf"] = lsq["x"]
        if "log_dog" in methods:
            lsq = least_squares(self.residual_log, p0, bounds=bounds,
                                method="dogbox", args=(r, rho, mass))
            dic["c_log_dog"] = lsq["x"]
        ################################################################################################
        if "Child18_lm" in methods:
            lsq = least_squares(self.residual_Child18, p0, method="lm",
                                args=(r, N_part_in_shell, size_shell, mass))
            dic["c_Child18_lm"] = lsq["x"]
        if "Child18_trf" in methods:
            lsq = least_squares(self.residual_Child18, p0, method="trf", bounds=bounds, 
                                args=(r, N_part_in_shell, size_shell, mass))
            dic["c_Child18_trf"] = lsq["x"]
        if "Child18_dog" in methods:
            lsq = least_squares(self.residual_Child18, p0, method="dogbox", bounds=bounds,
                                args=(r, N_part_in_shell, size_shell, mass))
            dic["c_Child18_dog"] = lsq["x"]
        ###############################################################################################
        if "Bhattacharya13_lm" in methods:
            lsq = least_squares(self.residual_Bhattacharya13, p0, method="lm",
                                args=(r_shell[1:], N_part_in_shell, mass, r_shell[0]))
            dic["c_Bhattacharya13_lm"] = lsq["x"]
        if "Bhattacharya13_trf" in methods:
            lsq = least_squares(self.residual_Bhattacharya13, p0, method="trf", bounds=bounds,
                                args=(r_shell[1:], N_part_in_shell, mass, r_shell[0]))
            dic["c_Bhattacharya13_trf"] = lsq["x"]
        if "Bhattacharya13_dog" in methods:
            lsq = least_squares(self.residual_Bhattacharya13, p0, method="dogbox", bounds=bounds,
                                args=(r_shell[1:], N_part_in_shell, mass, r_shell[0]))
            dic["c_Bhattacharya13_dog"] = lsq["x"]
        return dic
        
    def test_concentration(self, methods, N_part, N_bin, N_halos,):
        conc = pd.DataFrame()
        for h in range(N_halos):
            my_halo = self.smooth_halo_creation(N_part=N_part, N_bin=N_bin) #kind_profile, b_ax=0.5, c_ax=0.5)
            data = my_halo["data"]
            r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
            dic = self.fit_concentration(r_data=r_data, N_bin=N_bin, methods=methods)
            conc = pd.concat([conc, pd.DataFrame(dic, index=[h])])
        return conc


# the squared of res is minimized if loss='linear' (default) # otherwise, see
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares

# masses of a halo of mass mass and concentration conc, inside spheres of radius radius, according to NFW
# see equation 2 of Bhattacharya+13: https://iopscience.iop.org/article/10.1088/0004-637X/766/1/32/pdf

# it computes the residual for the fit, see the equation 5 of Child+18
# https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract

# https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable

if __name__ == '__main__':
    c_comp = Concentration_accumulation()
    N_part = 1000
    N_bin = 10
    N_halos = 1000
    box_arr = [1, 2, 3, 4, 5]
    wl_arr = [2, 3, 4, 5]
    pol_arr = [1, 2, 3, 4, 5]
    #c_acc, c_peak, dc_peak, c_peak_sg, dc_peak_sg = c_comp.test_many_computation(box_arr, wl_arr, pol_arr, 
    #                                                                             N_halos, N_part, N_bin)
    #c_comp.do_plot(c_acc, c_peak, dc_peak, c_peak_sg, dc_peak_sg)
    
    methods = ["log_trf", "log_dog",
               "Child18_trf","Child18_dog",
               "Bhattacharya13_trf", "Bhattacharya13_dog"]
    #conc = c_comp.test_concentration(methods, N_part=N_part, N_bin=N_bin, N_halos=N_halos)
    #print(conc.mean())
    #print(conc.std())
