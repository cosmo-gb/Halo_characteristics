#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 02 15:54:00 2023

@author: guillaume
"""


"""
To do: set the tests alone in a class and the concentration computation in another class

This script contains 1 class Concentration_computation
which allows to compute the concentration of a halo 
following multiple techniques:
- the mass accumulation technique (based on NFW, see https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf)
- peak finding (not based on NFW, see https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf)
- fitting method (different fitting methods are use):
    - lm
    - dog
    - trf: and each with different residuals:
        - Child18
        - Bathhachary13
        - log

It contains the following methods:

    - compute_c_NFW_acc: for accumulated mass technique

    - smooth: for peak finding technique
    - get_c_from_smooth_data: for peak finding technique

    - residual_log: for fitting technique
    - residual_Child18
    - residual_Bhattacharya13
    - find_bining

    - build_list: for many computations of concentration with different smoothing styles for peak finding technique
    - compute_many_c_smoothing: many computations of concentration with different smoothing styles for peak finding technique
    - test_many_computation: many computations of concentration with both accumulated mass and peak finding techniques
    - do_plot: plot the results of the test for both accumulated mass and peak finding
"""

# computation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# typing
from typing import Tuple, List, Dict
# stat
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

# my code
# Be carefull, you need to be in the correct folder 
# ~/Bureau/Cosmo_after_thesis/DEUS_halo/Codes/Halo_characteristics
# in order to run this line of code
from semi_analytical_halos.generate_smooth_halo import Smooth_halo



class Concentration_computation(Smooth_halo):

    # real computation for accumulated mass technique
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
        # then I smooth this mass function
        M_in_sphere_r_s_NFW = self.smooth(M_in_sphere_r_s_NFW, box_pts)
        # difference between the masses: M(<r) - M_NFW(<r_s=r)
        diff = M_in_sphere_data - M_in_sphere_r_s_NFW 
        # the mass difference changes of sign at r=r_s
        # thus we look for this change of sign
        ind_sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(ind_sign_change) == 1: # case with only 1 solution
            ok = True
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

    # real computation for peak finding technique
    def smooth(self, y: np.ndarray[float], box_pts: int,) -> np.ndarray[float]:
        """
        Computes a convolution of y in order to smooth y
        e.g. if box_pts=3 then y[i]  ->  (y[i-1] + y[i] + y[i+1])/(3)
        # inputs:
        - y: np.ndarray[float], time serie I want to smooth/convolve
        - box_pts: int, smoothing parameters
        # outputs:
        - y_smooth: np.ndarray[float], y smoothed
        """
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # real computation for peak finding technique
    def get_c_from_smooth_data(self, dn_dr_smooth: np.ndarray[float], r_log_bin: np.ndarray[float],
                               add_ind: int=0,) -> np.ndarray[float]:
        """
        Computes the concentration of a profile with the peak finding method
        i.e. r = r_{-2} where rho(r) * r**2 has its maximum 
        and concentration = 1/r_{-2} in halo radius unit
        This does NOT assume NFW or any analytical profile
        # inputs:
        - dn_dr_smooth: np.ndarray[float], rho(r) * r**2 smoothed
        - r_log_bin: np.ndarray[float], radius r
        - add_bin: int, default = 0, change the maximal position, this parameters should be removed
        # outputs:
        - np.array([[float, float]]), concentration and its rough error estimate
        """
        dn_dr_max = np.max(dn_dr_smooth)
        ind_max = np.where(dn_dr_smooth == dn_dr_max)[0] # where the maximum is 
        r_minus_2 = r_log_bin[ind_max+add_ind] # r_{-2}
        conc = 1/r_minus_2 # concentration in halo radius unit
        # then compute estimate of error
        r_minus_2_up = r_log_bin[ind_max+add_ind+1]
        r_minus_2_down = r_log_bin[ind_max+add_ind-1]
        conc_low = 1/r_minus_2_up
        conc_high = 1/r_minus_2_down
        dc = (conc_high - conc_low)/2
        return np.array([conc, dc]).T
    
    # test for peak finding technique
    def build_list(self, wl_arr: List[int], pol_arr: List[int],) -> List[List[int]]:
        """
        for many computations of the concentration with many smoothing styles 
        for the savgol_filter method
        # inputs:
        - wl_arr: List[int], smoothing parameter
        - pol_arr: List[int], smoothing parameter
        # output:
        - my_list: List[List[int]], smoothing parameters
        """
        my_list = []
        for w in wl_arr:
            for p in pol_arr:
                if p < w:
                    my_list += [[w,p]]
        return my_list
    
    # test for peak finding technique
    def compute_many_c_smoothing(self, r_data: np.ndarray[float], N_bin: int,
                                 box_arr: List[int], wl_pol: List[List[int]],) \
                                 -> Tuple[np.ndarray[float]]:
        """
        run the get_c_from_smooth_data method in order to compute the concentration
        for 2 kinds of smoothing methods: smooth method and savgol_filter from scipy.signal
        each methods is tested for different smoothing parameters 
        (box and wl for smooth and savgol_filter respectively)
        # inputs:
        - r_data: np.ndarray[float], radius of the particles inside a halo in halo radius unit
        - N_bin: int, number of logarithmic bins to compute the density profile
        - box_arr: List[int], parameters for smoothing the smooth method
        - wl_pol: List[int], parameters for smoothing with the savgol_filter method
        # outputs:
        - c_peak: np.ndarra[float], concentration obtained with the smooth method
        - dc_peak: np.ndarra[float], estimate of error
        - c_peak_sg: np.ndarray[float], concentrations obtained with the savgol_filter method
        - dc_peak_sg: np.ndarra[float], estimate of error
        """
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
    
    # test for both peak finding and accumulated mass techniques
    def test_many_computation(self, box_arr: List[int], wl_arr: List[int], pol_arr: List[int],
                              N_halos: int, N_part: int, N_bin: int,) -> Tuple[pd.DataFrame]:
        """
        Tests many computation methods of concentration based on accumulated mass and peak finding.
        # inputs:
        - box_arr: List[int], parameters of smoothing for smooth method
        - wl_arr: List[int], parameters of smoothing for savgol_filter method
        - pol_arr: List[int], parameters of smoothing for savgol_filter method
        - N_halos: int, number of halos created for the test
        - N_part: int, number of particles inside each halo created
        - N_bin: int, number of bin for the profile of the created halo
        # outputs: pd.DataFrame of N_halos rows and N_methods columns
        - c_acc: pd.DataFrame, concentration computed with the accumulated method
        - c_peak: pd.DataFrame, concentration obtained with the smooth method
        - dc_peak: pd.DataFrame, estimate of error
        - c_peak_sg: pd.DataFrame, concentrations obtained with the savgol_filter method
        - dc_peak_sg: pd.DataFrame, estimate of error
        """
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
            #################################################################################
            # accumulated mass computation method
            c_acc[t], ok = self.compute_c_NFW_acc(r_data)
            if ok == False:
                print(ok, c_acc[t])
            ##################################################################################
            # smoothingcomputation method
            cs = self.compute_many_c_smoothing(r_data, N_bin, box_arr, wl_pol)
            c_peak[t:t+1], dc_peak[t:t+1] = cs[0], cs[1]
            c_peak_sg[t:t+1], dc_peak_sg[t:t+1] = cs[2], cs[3]
        ##################################################################################
        # rearangement in dataframes
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
    
    # plot of the results of the test
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

    # real computation for fitting technique
    def residual_log(self, param: np.ndarray[float], x_data: np.ndarray[float],
                     y_data: np.ndarray[float], mass: float,) -> np.ndarray[float]:
        """
        Computes the residual of the analytical density profile wrt the profile of the data
        # inputs:
        - param: np.ndarray[float], len=1, contains the concentration of the analytical profile
        - x_data: np.ndarray[float], len=N_bin, radius in the halo size unit
        - y_data: np.ndarray[float], len=N_bin, density profile
        - mass: float, halo mass
        # outputs:
        - residual: np.ndarray[float], len=N_bin
        """
        y_th = self.profile_NFW(x_data, param, mass)
        # the squared of res is minimized if loss='linear' (default)
        return np.log10(y_data) - np.log10(y_th)
    
    # real computation for fitting techniques
    def residual_Child18(self, param: np.ndarray[float], x_data: np.ndarray[float],
                         dn: np.ndarray[float], dr: np.ndarray[float],
                         mass: float,)  -> np.ndarray[float]:
        """
        Computes the residual of the analytical density profile wrt the profile of the data
        # inputs:
        - param: np.ndarray[float], len=1, contains the concentration of the analytical profile
        - x_data: np.ndarray[float], len=N_bin, radius in the halo size unit
        - dn: np.ndarray[float], len=N_bin, number of particles in bins
        - dr: np.ndarray[float], len=N_bin, bin size
        - mass: float, halo mass
        # outputs:
        - residual: np.ndarray[float], len=N_bin
        """
        # it computes the residual for the fit, see the equation 5 of Child+18
        # https://ui.adsabs.harvard.edu/abs/2018ApJ...859...55C/abstract
        rho_NFW = self.profile_NFW(x_data, param, mass)
        dn_dr_NFW = 4 * np.pi * x_data**2 * rho_NFW
        # the squared of residual is minimized if the option loss='linear' (default) in least_squares
        return np.sqrt(((dn/dr - dn_dr_NFW)**2)/(dn/dr**2)) 
    
    # real computation for fitting techniques 
    def residual_Bhattacharya13(self, param: np.ndarray[float], x_data: np.ndarray[float],
                                dn: np.ndarray[float], mass: float,
                                r_min: float,) -> np.ndarray[float]:
        """
        Computes the residual of the analytical density profile wrt the profile of the data
        # inputs:
        - param: np.ndarray[float], len=1, contains the concentration of the analytical profile
        - x_data: np.ndarray[float], len=N_bin, radius in the halo size unit
        - dn: np.ndarray[float], len=N_bin, number of particles in bins
        - mass: float, halo mass
        - r_min: float, halo resolution
        # outputs:
        - residual: np.ndarray[float], len=N_bin
        """
        # it computes the residual of the fit, see the equation 4 of Bhattacharya+13
        # https://iopscience.iop.org/article/10.1088/0004-637X/766/1/32/pdf
        M_in_sphere_r, M_shell_r = self.mass_NFW(x_data, param, mass, r_min)
        return ((dn - M_shell_r)**2)/dn
    
    # real computation for fitting techniques
    def find_bining(self, r_data: np.ndarray[float], N_bin: int,
                    N_min: int=1,) -> Tuple[Dict[str, np.ndarray], int]:
        """
        looks for the first bin number with at least N_min particles inside each bin
        # inputs:
        - r_data: np.ndarray[float], radius of each particle in halo size unit
        - N_bin: int, number of number of bin tested
        - N_min: int, default=1, minimum number of particle inside each bin
        # outputs:
        - out: Dict[str, np;ndarray], see profile_log_r_bin_hist method in density_profile
        - N_bin_use: int, number of bins that you should use
        """
        for b in range(1, N_bin):
            # decrease the binning
            N_bin_use = N_bin - b
            out = self.profile_log_r_bin_hist(r_data, N_bin=N_bin_use)
            N_part_in_shell = out["N_part_in_shell"]
            if np.min(N_part_in_shell) >= N_min:
                return out, N_bin_use
            
    # test of fitting technique
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
    
    # test of fitting technique       
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
    c_comp = Concentration_computation()
    N_part = 1000
    N_bin = 10
    N_halos = 100
    box_arr = [1, 2, 3, 4, 5]
    wl_arr = [2, 3, 4, 5]
    pol_arr = [1, 2, 3, 4, 5]
    c_acc, c_peak, dc_peak, c_peak_sg, dc_peak_sg = c_comp.test_many_computation(box_arr, wl_arr, pol_arr, 
                                                                                 N_halos, N_part, N_bin)
    c_comp.do_plot(c_acc, c_peak, dc_peak, c_peak_sg, dc_peak_sg)
    
    methods = ["log_trf", "log_dog",
               "Child18_trf","Child18_dog",
               "Bhattacharya13_trf", "Bhattacharya13_dog"]
    conc = c_comp.test_concentration(methods, N_part=N_part, N_bin=N_bin, N_halos=N_halos)
    print(conc.mean())
    print(conc.std())
