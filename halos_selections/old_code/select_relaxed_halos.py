#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:18:47 2021

@author: guillaume
"""
import numpy as np
import random


#path_input_MF = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/input_MF_1000_part/relaxed_halos/DEUS/'
#path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/properties/'
#path_data = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/sphere_sel_1_Rvir/'
path_input_MF = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/input_MF_1000_part/relaxed_halos/DEUS/'
path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/properties/'
path_data = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/sphere_sel_1_Rvir/'
shift = np.loadtxt(path+'shift_all.dat')
prop = np.loadtxt(path+'prop_all.dat')
Rvir = prop[:,0]
Nvir = prop[:,1]
cdm_fof = prop[:,2:5]
cdm_dens = prop[:,5:9]
names = np.loadtxt(path+'names_all.dat',dtype=str)
s = np.sqrt( (cdm_fof[:,0] - cdm_dens[:,0])**2 + (cdm_fof[:,1] - cdm_dens[:,1])**2 +  (cdm_fof[:,2] - cdm_dens[:,2])**2 )
s /= Rvir
N_halos = 2
print(shift[0:N_halos])
print(s-shift)
s_th = 0.07
ind_relaxed = np.where(shift < s_th)[0]
print(len(ind_relaxed))
names_relaxed = names[ind_relaxed]
prop_relaxed = prop[ind_relaxed]
shift_relaxed = shift[ind_relaxed]
#np.savetxt(path+'relaxed_halos/prop_relaxed.dat',prop_relaxed)
#np.savetxt(path+'relaxed_halos/shift_relaxed.dat',shift_relaxed)
#np.savetxt(path+'relaxed_halos/names_relaxed.dat',names_relaxed,fmt='%s')
names_relaxed = np.loadtxt(path+'relaxed_halos/names_relaxed_b_good.dat',dtype=str)
shift_relaxed = np.loadtxt(path+'relaxed_halos/shift_relaxed_b_good.dat')
prop_relaxed = np.loadtxt(path+'relaxed_halos/prop_relaxed_b_good.dat')
abg_relaxed = np.loadtxt(path+'relaxed_halos/abg_relaxed_b_good.dat')
abc_relaxed = np.loadtxt(path+'relaxed_halos/abc_relaxed_b_good.dat')
Rvir = prop_relaxed[:,0]


#np.savetxt(path+'relaxed_halos/abc_relaxed_b_good.dat',abc_b_good)
#np.savetxt(path+'relaxed_halos/abg_relaxed_b_good.dat',abg_b_good)
#np.savetxt(path+'relaxed_halos/prop_relaxed_b_good.dat',prop_b_good)
#np.savetxt(path+'relaxed_halos/shift_relaxed_b_good.dat',shift_b_good)
#np.savetxt(path+'relaxed_halos/names_relaxed_b_good.dat',names_b_good,fmt='%s')


Rvir = prop_relaxed[:,0]
print(len(Rvir))
print(len(names_relaxed))
print(len(shift_relaxed))
Nvir = prop_relaxed[:,1]
cdm_fof = prop_relaxed[:,2:5]
cdm_dens = prop_relaxed[:,5:9]
my_push = 0.4
N_part_MF = 1000
N_times = 100
for h in range(len(Rvir)) :
    print(h)
    name = names_relaxed[h]
    my_name = name.replace('.dat','_sphere_sel_1_Rvir.dat')
    pos = np.loadtxt(path_data+my_name)
    #print(pos)
    #cdm_mean = np.mean(pos,axis=0)
    #print(cdm_mean)
    #shift_mean = np.sqrt(cdm_mean[0]**2 + cdm_mean[1]**2 + cdm_mean[2]**2 )
    #print(shift_mean/Rvir[h])
    '''if shift_mean/Rvir[h] > shift_relaxed[h] :
        print(h)
        print(shift_mean/Rvir[h])
        print(shift_relaxed[h])'''
    pos /= Rvir[h]
    pos += (1+my_push)
    N_part_tot = len(pos)
    my_name =  my_name.replace('.dat','')
    for n in range(N_times) :
        my_int = random.sample(range(N_part_tot),N_part_MF)
        pos_MF = pos[my_int]
        np.savetxt(path_input_MF+my_name+'_'+str(n)+'.res',pos_MF)