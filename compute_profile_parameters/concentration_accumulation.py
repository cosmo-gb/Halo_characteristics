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

from semi_analytical_halos.generate_smooth_halo import Smooth_halo


class Concentration_accumulation():

    def NFW_function(self, x: np.ndarray[float]) -> np.ndarray[float]:
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

    def mass_in_r_s_NFW(self, Mass_total: float, r_s: np.ndarray[float]) -> np.ndarray[float]:
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
        
    def compute_c_NFW_acc(self, r_data: np.ndarray[float], do_plot: bool=False) -> Tuple[float, bool]:
        """
        This function computes the concentration of a halo with the accumulated mass technique.
        It follows the equation 6 of https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
        #############################################################################
        Input parameters:
        - r_data: np.array of float, radius of particles in the halo
        - do_plot: bool, default is False, plot or not the masses M(<r)
        #############################################################################
        Returns:
        - concentration: float, the halo concentration
        - ok: bool, if True there is only 1 unique solution, False otherwise
        """
        # compute the concentration with accumulation mass method
        # following: https://iopscience.iop.org/article/10.3847/1538-4357/aabf95/pdf
        # it assumes NFW
        r_data_sorted = np.sort(r_data) # I need to sort the radii
        N_part_tot = len(r_data_sorted) # total number of particles
        N_end = int(0.9 * N_part_tot) # considered part of the halo
        M_in_sphere_data = range(1, N_end+1) # cumulative mass
        # then I compute the mass within r_s, assuming that r=r_s for all r
        M_in_sphere_r_s_NFW = self.mass_in_r_s_NFW(N_part_tot, r_data_sorted[0:N_end])
        diff = M_in_sphere_data - M_in_sphere_r_s_NFW # difference between the mass
        # when the mass difference changes of sign, it means that we have r=r_s
        ind_sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(ind_sign_change) == 1: # case with only 1 solution
            ok = True
            r_s = (r_data_sorted[ind_sign_change[0]] + r_data_sorted[ind_sign_change[0] - 1])/2
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
            


if __name__ == '__main__':
    halo = Smooth_halo()
    kind_profile = {"kind of profile": "abg",
                    "concentration": 10,
                    "alpha": 1, "beta": 3, "gamma": 1}
    my_halo = halo.smooth_halo_creation() #kind_profile, b_ax=0.5, c_ax=0.5)
    data = my_halo["data"]
    print("N_tot =",my_halo["N_tot"])
    #halo.plot_data(data[:,0],data[:,1])
    halo.beauty_plot_colorbar(data)

    r_data = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    c_acc = Concentration_accumulation()
    conc = c_acc.compute_c_NFW_acc(r_data)
    print(conc)