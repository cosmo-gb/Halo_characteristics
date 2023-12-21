#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:12:23 2021

@author: guillaume
"""
import numpy as np
import pandas as pd
from random import sample
import sys


cluster = False
if cluster == True : # case where I apply this code on the cluster
    cosmo = str(sys.argv[1])
    z_str = str(sys.argv[2])
    mass_str = str(sys.argv[3])
    N_haloes = 256
    save_file = True
    #n_parallel = np.int(sys.argv[4]) # I import a number from a terminal command line
else : # case where I apply this code on my laptop
    cosmo = 'lcdm'
    z_str = '0'
    #mass_str = 'most_massive'
    mass_str = 'mass_10_14_5_Msun_beyond_FOF'
    N_haloes = 247
    save_file = False
    #n_parallel = 128
    
    
file_param_name = 'param_DEUS_box_648_'+cosmo+'w7_z_'+z_str
file_param = __import__(file_param_name) 

L_box = file_param.L_box
rho_vir_Brian_Norman = file_param.rho_vir_Brian_Norman
mass_one_particle = file_param.mass_one_particle
rho_crit_0 = file_param.rho_crit_0
Omega_matter_0 = file_param.Omega_matter_0
redshift = file_param.redshift
v_ren = file_param.v_ren
hsml = file_param.hsml


def periodicity(x,y,z,cdm,dis_1=0.5,dis_2=0.9,box=1):
    ''' This function checks if there is a periodicity problem with the data,
    because particles could lie at the border of the simulation box.
    So for e.g. a particle p1 in such halo could have a position x[p1]=0.01
    and another particle p2 in the same halo x[p2]=1.01 In the simulation 
    those 2 particles are considered at a distance of 0.02 (periodicity condition)
    x, y, z contains the particle positions in box unit.
    cdm is the halo center in box unit (highest density point)
    dis_1 is the distance is the threshold limit distance that I set
    so if 2 particles are at a smaller distance than dis_1 in 1 direction, 
    then I consider that there is no problem, by default = 0.5
    dis_2 is the distance in one direction, which allows me to test 
    if the particle p is close to 0 or 1, by default = 0.9
    box is the length of the box simulation, by default = 1
    It returns the particle positions without periodicity problem
    and bad_periodicity=0 if there was no problem and bad_periodicity=1
    if the problem had been removed.
    '''
    distance = np.sqrt((x-cdm[0])**2 + (y-cdm[1])**2 + (z-cdm[2])**2) # distance from the center in box unit
    indice_problem = np.where(distance > dis_1)[0] # where the distance is VERY large
    if len(indice_problem) > 0 :
        bad_periodicity = 1 # there is a periodicity problem
    else :
        bad_periodicity = 0 # there is no periodicity problem
    if bad_periodicity == 1 : # then I correct the problem
        # I deal with the problem in the 3 direction of space separetely
        distance_x, distance_y, distance_z = np.abs(x - cdm[0]), np.abs(y - cdm[1]), np.abs(z - cdm[2])
        ind_x, ind_y, ind_z = np.where(distance_x > dis_1)[0], np.where(distance_y > dis_1)[0], np.where(distance_z > dis_1)[0]
        my_x, my_y, my_z = x[ind_x], y[ind_y], z[ind_z] # contains the problematic positions
        N_x, N_y, N_z = len(my_x), len(my_y), len(my_z) # number of problematic positions
        # I first check if the problematic positions are > dis_2 (thought as close to 1), 
        #and I remove box for those where it is the case.
        # Then I just add 1 for the problematic case which are not > dis_2
        ind_x_minus, ind_y_minus, ind_z_minus = np.where(my_x > dis_2)[0], np.where(my_y > dis_2)[0], np.where(my_z > dis_2)[0]
        my_x[ind_x_minus], my_y[ind_y_minus], my_z[ind_z_minus] = my_x[ind_x_minus] - box, my_y[ind_y_minus] - box, my_z[ind_z_minus] - box
        ind_x_plus, ind_y_plus, ind_z_plus = np.ones(N_x,dtype=bool), np.ones(N_y,dtype=bool), np.ones(N_z,dtype=bool)
        ind_x_plus[ind_x_minus], ind_y_plus[ind_y_minus], ind_z_plus[ind_z_minus] = False, False, False
        my_x[ind_x_plus], my_y[ind_y_plus], my_z[ind_z_plus] = my_x[ind_x_plus] + box, my_y[ind_y_plus] + box, my_z[ind_z_plus] + box
        x[ind_x], y[ind_y], z[ind_z] = my_x, my_y, my_z # I replace the corrected particles
                    
    return(x,y,z,bad_periodicity)

def select_samples_for_MF(cosmo='lcdm',mass_str=mass_str,z_str='0',
                          N_haloes=2,N_samples=30,N_part_MF=1000,my_push=0.4) :
    #path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+mass_str+'.5_Msun_beyond_FOF/'
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
    #path_MF = '../../../../DEUSS_648Mpc_2048_particles/input_MF/cosmo_DE_z_impact/z_0/'+cosmo+'/mass_Mvir_10_'+mass_str+'_5/'
    path_MF = '../../../../DEUSS_648Mpc_2048_particles/input_MF/cosmo_DE_z_impact/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
    names = np.array(pd.read_csv(path+'halo_properties/names_sel.dat'))[:,0]
    prop = np.array(pd.read_csv(path+'halo_properties/properties_numbers_sel.dat'))
    #print(names)
    #print(prop)
    Rvir_all, Nvir_all = prop[:,0], np.array(prop[:,1],dtype=int)
    cdm_all = prop[:,5:8]
    #print(Rvir_all)
    for h in range(N_haloes,N_haloes+1) : #range(N_haloes) :
        #fof_boxlen648_n2048_wcdmw7_strct_00002_00735_sphere_1_Rvir.dat
        name_h = names[h]
        name_h = name_h.replace('.dat','')
        print(name_h)
        print('Lbox =',L_box)
        data = np.array(pd.read_csv(path+'data/'+name_h+'_sphere_1_Rvir.dat',sep=',', comment='#'))
        print(data)
        pos = data[:,0:3]
        cdm_h = cdm_all[h]
        print(pos)
        print(cdm_h)
        Rvir_h, Nvir_h = Rvir_all[h], Nvir_all[h]
        print(Rvir_h)
        if cosmo != 'lcdm' or mass_str == 'most_massive' :
            x, y, z, bad_periodicity = periodicity(pos[:,0]/L_box,pos[:,1]/L_box,pos[:,2]/L_box,cdm_h/L_box)
            print('bad_periodicity =',bad_periodicity)
            pos = np.zeros((Nvir_h,3))
            pos[:,0], pos[:,1], pos[:,2] = x*L_box, y*L_box, z*L_box
            pos = pos - cdm_h
        print(pos)
        pos = pos/Rvir_h
        print(pos)
        pos = pos + 1 + my_push
        print(pos)
        ind_pb = np.where( (pos < my_push) | (pos > 2 + my_push))
        if len(ind_pb[0]) != 0 :
            print('Problem of particle positions')
            print(ind_pb)
            print(pos)
        for s in range(N_samples) :
            int_s = sample(range(Nvir_h),N_part_MF)
            pos_s = pos[int_s]
            if s == 0 :
                print(pos_s)
            #np.savetxt(path_MF+name_h+'_sphere_1_Rvir_'+str(s)+'.dat',pos_s)
            np.savetxt(path_MF+name_h+'_sphere_1_Rvir_in_'+str(s)+'.dat',pos_s)
    #print(data)
    return()


select_samples_for_MF(cosmo=cosmo,z_str=z_str,mass_str=mass_str,N_haloes=N_haloes)
        


         