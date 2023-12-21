#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:01:30 2021

@author: guillaume
"""
import numpy as np
import random

if __name__ == '__main__' :
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_14_5_sphere_sel/properties/'
    path_haloes = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/sphere_sel_1_Rvir/'
    path_saved = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_14_5_sphere_sel/data_MF/DEUS/'
    prop = np.loadtxt(path+'prop_all.dat')
    names_all = np.genfromtxt(path+'names_all.dat',dtype=str)
    indices = np.loadtxt(path+'indices_non_shifted_haloes.dat')
    indices = np.array(indices,dtype=int)
    print(indices[0:10])
    prop = prop[indices]
    print(names_all[0:10])
    names_all = names_all[indices]
    print(names_all[0:10])
    Rvir = prop[:,0]
    N_haloes = len(indices)
    N_samples = 30
    my_push = 1.4
    N_part_MF = 1000
    for h in range(N_haloes) :
        #names = 'fof_boxlen648_n2048_lcdmw7_strct_00002_02733_sphere_sel_1_Rvir.dat'
        name = names_all[h]
        name_used = name.replace('.dat', '_sphere_sel_1_Rvir.dat')
        print(name_used)
        pos = np.loadtxt(path_haloes+name_used)
        N_vir = len(pos)
        pos /= Rvir[h]
        pos += my_push
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        #radius = np.sqrt(x**2 + y**2 + z**2)
        #print(np.min(radius), np.max(radius))
        print(np.min(x),np.max(x))
        name_saved = name.replace('.dat', '_N_part_MF_1000_')
        for s in range(N_samples) :
            my_integers = random.sample(range(N_vir),N_part_MF) # sampling
            pos_sample = pos[my_integers]
            np.savetxt(path_saved+name_saved+str(s)+'.dat',pos_sample)
            
        
        
