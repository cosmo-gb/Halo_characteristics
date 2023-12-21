#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:45:22 2022

@author: guillaume
"""


'''
This code checks if twins halos are fine and it also concatenates their indices according to the cosmo or the redshift
'''

import numpy as np

N_halos = 256
N_red, N_cosmo = 5, 3
mass_str = 'most_massive'
path_0 = '../../../../DEUSS_648Mpc_2048_particles/'
twins = np.zeros((N_red,N_cosmo,N_halos),dtype=int)
twins[:,0] = np.array(range(N_halos))
z_array = ['0','0_5','1','1_5','2']
cosmo_array = ['lcdm','rpcdm','wcdm']
z = 0
for z_str in z_array :
    c = 1
    for cosmo in ['rpcdm','wcdm'] :
        path_end = '/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
        path_prop = path_0+'mass_bin_data'+path_end+'halo_properties/'
        twins[z,c] = np.loadtxt(path_prop+'twins_halo_cosmo.dat')[:,0]
        ind_ok = np.where(twins[z,c] != -1)[0]
        #print('z =',z_str,'cosmo =',cosmo,'frac_twins =',len(ind_ok)/N_halos)
        #print(twins)
        #print(twins_old)
        #twins_eq = np.array_equal(twins,twins_old)
        #print(twins_eq)
        c += 1
        #break
    #break
    z += 1
print(twins)
###################################################################################"
ind_twins_cosmo = np.ones((N_red,N_halos,N_cosmo),dtype=int) * (-1)
z = 0
cosmo = 'lcdm'
for z_str in z_array :
    break
    ind_ok = np.where( (twins[z,1] != -1) & (twins[z,2] != -1) )[0]
    N_ok = len(ind_ok)
    ind_twins_cosmo[z,0:N_ok,0] = ind_ok
    ind_twins_cosmo[z,0:N_ok,1] = twins[z,1,ind_ok]
    ind_twins_cosmo[z,0:N_ok,2] = twins[z,2,ind_ok]
    print(N_ok)
    path_end = '/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
    path_prop = path_0+'mass_bin_data'+path_end+'halo_properties/'
    file_save = open(path_prop+'twins_halos_cosmo_concatenate.dat', 'w')
    file_save.write('# identity of the halos wich are the twins (if it does not exist =-1) of the lcdm halos \n')
    file_save.write('# L_box=648Mpc/h, z = '+z_str+', 256 '+mass_str+' halos, with Brian&Norman98 selection \n')
    file_save.write('# halo identity lcdm, halo identity rpcdm, halo identity wcdm  \n')
    np.savetxt(file_save,ind_twins_cosmo[z])
    file_save.close()
    z += 1
print(ind_twins_cosmo)
############################################################################################
twins = np.zeros((N_cosmo,N_red,N_halos),dtype=int)
twins[:,0] = np.array(range(N_halos))
c = 0
for cosmo in cosmo_array :
    z = 1
    for z_str in ['0_5','1','1_5','2'] :
        path_end = '/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
        path_prop = path_0+'mass_bin_data'+path_end+'halo_properties/'
        twins[c,z] = np.loadtxt(path_prop+'twins_halo_redshift.dat')[:,0]
        #ind_ok = np.where(twins != -1)[0]
        #print('z =',z_str,'cosmo =',cosmo,'frac_twins =',len(ind_ok)/N_halos)
        z += 1
    c += 1
####################################################################################
ind_twins_redshift = np.ones((N_cosmo,N_halos,N_red),dtype=int) * (-1)
c = 0
z_str = '0'
for cosmo in cosmo_array :
    break
    ind_ok = np.where( (twins[c,1] != -1) & (twins[c,2] != -1) & (twins[c,3] != -1) & (twins[c,4] != -1) )[0]
    N_ok = len(ind_ok)
    ind_twins_redshift[c,0:N_ok,0] = ind_ok
    ind_twins_redshift[c,0:N_ok,1] = twins[c,1,ind_ok]
    ind_twins_redshift[c,0:N_ok,2] = twins[c,2,ind_ok]
    ind_twins_redshift[c,0:N_ok,3] = twins[c,3,ind_ok]
    ind_twins_redshift[c,0:N_ok,4] = twins[c,4,ind_ok]
    print(N_ok)
    path_end = '/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
    path_prop = path_0+'mass_bin_data'+path_end+'halo_properties/'
    file_save = open(path_prop+'twins_halos_redshift_concatenate.dat', 'w')
    file_save.write('# identity of the halos wich are the twins (if it does not exist =-1) of the z=0 halos \n')
    file_save.write('# L_box=648Mpc/h, '+cosmo+'w7, 256 '+mass_str+' halos, with Brian&Norman98 selection \n')
    file_save.write('# halo identity z=0, halo identity z=0.5, halo identity z=1, halo identity z=1.5, halo identity z=2  \n')
    np.savetxt(file_save,ind_twins_redshift[c])
    file_save.close()
    c += 1