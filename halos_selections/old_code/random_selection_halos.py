#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:06:56 2021

@author: guillaume
"""

import numpy as np
from random import sample
import os


def random_sel_halos(path,N_halos_tot,N_sel):
    # random selection of haloes
    my_int = sample(range(N_halos_tot),N_sel)
    print(my_int)
    my_int = np.sort(my_int)
    print(my_int)
    names_all = np.genfromtxt(''+path+'names_all.dat',dtype=str) # names
    names_sel = names_all[my_int]
    print(names_sel)
    np.savetxt(''+path+'names_sel_random.dat',names_sel,fmt="%s")
    prop_all = np.loadtxt(''+path+'prop_all.dat') # prop (Rvir, cdms)
    prop_sel = prop_all[my_int]
    np.savetxt(''+path+'prop_sel_random.dat',prop_sel)
    cdm_all = np.loadtxt(''+path+'cdm_and_shift_all.dat') # cdm shift
    cdm_sel = cdm_all[my_int]
    np.savetxt(''+path+'cdm_and_shift_sel_random.dat',cdm_sel)
    q_all = np.loadtxt(''+path+'equilibrium_all.dat') # q
    q_sel = q_all[my_int]
    np.savetxt(''+path+'equilibrium_sel_random.dat',q_sel)
    shape_all = np.loadtxt(''+path+'abc_sphericity_CV_all.dat') # shape
    shape_sel = shape_all[my_int]
    np.savetxt(''+path+'abc_sphericity_CV_sel_random.dat',shape_sel)
    return()

def load_data_in_cluster(path_name,path_cluster,path_my_laptop,N_sel):
    names_sel = np.genfromtxt(''+path_name+'names_sel_random.dat',dtype=str)
    for s in range(N_sel) :
        print(s)
        name = names_sel[s]
        name_pdf = ''+name+'_5_Rvir_new.dat'
        my_command = 'rsync -av gbonnet@cc.lam.fr:'+path_cluster+''+name_pdf+' '+path_my_laptop+''
        print(my_command)
        os.popen(my_command).read()
    return()




if __name__ == '__main__' :
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/halo_properties/cluster_3/'
    N_halos_tot = 328
    N_sel = 30
    #random_sel_halos(path,N_halos_tot,N_sel)
    path_cluster = '/net/GECO/users/gbonnet/work/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/'
    path_my_laptop = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/'
    N_sel = 30
    #load_data_in_cluster(path,path_cluster,path_my_laptop,N_sel)
