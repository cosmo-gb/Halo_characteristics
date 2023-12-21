#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 18:14:43 2021

@author: guillaume
"""

import numpy as np
import time
import sys
path = '../DEUS_halo/' # path to the code I want to import
sys.path.append(''+path+'')
from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import L_box, rho_vir_Brian_Norman, mass_one_particle


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

def spherical_selection(pos,Rvir=1,cdm=np.array((0,0,0))) :
    ''' This functions does a spherical selection of pos
    '''
    x, y, z = pos[:,0]-cdm[0], pos[:,1]-cdm[1], pos[:,2]-cdm[2]
    radius = np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
    ind_sphere = np.where(radius <= Rvir)[0]
    r_sphere = radius[ind_sphere]
    pos_sphere = pos[ind_sphere]
    return(pos_sphere,r_sphere)

if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/'
    path_to_prop = path+'halo_properties/'
    path_to_data = path+'data/'
    names = np.loadtxt(path_to_prop+'names_all.dat',dtype=str)
    prop = np.loadtxt(path_to_prop+'prop_all.dat')
    Rvir = prop[:,0]
    Nvir = prop[:,1]
    cdm_fof = prop[:,2:5]
    cdm_dens = prop[:,5:]
    shift = np.sqrt( np.power(cdm_dens[:,0]-cdm_fof[:,0],2) + np.power(cdm_dens[:,1]-cdm_fof[:,1],2) + np.power(cdm_dens[:,2]-cdm_fof[:,2],2) )
    shift /= Rvir
    print(len(shift))
    #np.savetxt(path_to_prop+'shift_all.dat',shift)
    N_halos = 0
    t0 = time.time()
    for h in range(N_halos) :
        name = names[h].replace('.dat','.dat_5_Rvir_new.dat')
        #fof_boxlen648_n2048_lcdmw7_strct_00511_02561.dat_5_Rvir_new.dat
        data = np.loadtxt(path_to_data+name)
        x, y, z = data[0], data[1], data[2]
        print(x,y,z)
        print(cdm_dens[h])
        x,y,z,bad_periodicity = periodicity(x,y,z,cdm_dens[h]/L_box)
        N_part_tot = len(x)
        pos = np.zeros((N_part_tot,3))
        pos[:,0], pos[:,1] , pos[:,2]  = x, y, z
        pos *= L_box
        pos -= cdm_dens[h]
        print(pos[0:10])
        print(Rvir[h])
        pos_sphere,r_sphere = spherical_selection(pos,Rvir=Rvir[h])
        name = name.replace('.dat_5_Rvir_new.dat','_sphere_sel_1_Rvir.dat')
        np.savetxt(path_to_data+'sphere_sel_1_Rvir/'+name,pos_sphere)
    t1 = time.time()
    print(t1-t0)
        