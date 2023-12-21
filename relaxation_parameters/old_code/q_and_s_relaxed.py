#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 06:55:15 2022

@author: guillaume
"""

'''
This code computes the relaxation state of haloes of DEUS.
It computes the shift and the virial parameter q=2*K/V
'''

import numpy as np
import time
import pandas as pd
import sys
import unsiotools.simulations.cfalcon as falcon
cf=falcon.CFalcon()

cluster = False
if cluster == True : # case where I apply this code on the cluster
    # for the cluster array run:
    #0,1,2 <=> lcdm, mass=12,13,14
    #3,4,5 <=> rpcdm, mass=12,13,14
    #6,7,8 <=> wcdm, mass=12,13,14
    num_cosmo_mass = np.int(sys.argv[1])
    if num_cosmo_mass < 3 :
        cosmo = 'lcdm'
    elif num_cosmo_mass < 6 :
        cosmo = 'rpcdm'
    else :
        cosmo = 'wcdm'
    reste = num_cosmo_mass % 3
    if reste == 0 :
        mass_str_num = '12_5'
    elif reste == 1 :
        mass_str_num = '13_5'
    else :
        mass_str_num = '14_5'
    z_str = '0'
    N_haloes = 2
    save_file = True
    do_complete = True
else : # case where I apply this code on my laptop
    cosmo = 'lcdm'
    z_str = '0'
    mass_str_num = '13_5'
    N_haloes = 2
    save_file = True
    do_complete = False
    
mass_str = 'mass_10_'+mass_str_num+'_Msun_beyond_FOF'       
file_param_name = 'param_DEUS_box_648_'+cosmo+'w7_z_'+z_str
file_param = __import__(file_param_name) 

L_box = file_param.L_box
rho_vir_Brian_Norman = file_param.rho_vir_Brian_Norman
mass_one_particle = file_param.mass_one_particle
rho_crit_0 = file_param.rho_crit_0
Omega_matter_0 = file_param.Omega_matter_0
redshift = file_param.redshift
v_ren = file_param.v_ren
hsml = file_param.hsml * L_box/648
G_Newton = file_param.G_Newton


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

def put_in_frame(pos,Nvir,cdm,box):
    x, y, z = pos[:,0]/box, pos[:,1]/box, pos[:,2]/box
    x, y, z, bad_periodicity = periodicity(x,y,z,cdm/box)
    pos_frame = np.zeros((Nvir,3))
    pos_frame[:,0], pos_frame[:,1], pos_frame[:,2] = x*box, y*box, z*box
    pos_frame -= cdm
    return(pos_frame)

def compute_q_approx(pos,K_tot,N_part):
    ''' It computes the virial ratio approximatively
    by using an approximation on the gravitational field (spherical symmetry)
    pos are the particle positions (Mpc/h)
    K_tot are the total kinetic energy
    N_part is the number of particles
    '''
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    ind_r = np.argsort(radius)
    r_sorted = radius[ind_r]
    mass = np.linspace(1,N_part,N_part)
    V_all = -mass/r_sorted
    V_tot = np.sum(V_all) * G_Newton
    q = 2*K_tot/np.abs(V_tot)
    return(q)

def kinetic(vel):
    vx, vy, vz = vel[:,0], vel[:,1], vel[:,2]
    K_all = (vx**2 + vy**2 + vz**2)/2
    K_tot = np.sum(K_all)
    return(K_tot)

def compute_q(pos,K_tot,N_part):
    ''' This function computes the virial ratio 
    pos is the particle position in Mpc/h 
    it should be a float32 array of 3*N_part dimension (Dehnen function)
    vel is the particle velocities in km/s
    N_part is the number of particle
    '''
    mass_array = np.ones((N_part),dtype=np.float32)
    ok, acc, phi = cf.getGravity(pos, mass_array, hsml, 
                                       G=1.0, theta=0.6, kernel_type=1, ncrit=6)
    # I reset the potential gravitational energy in physical unit
    V_all = phi * G_Newton
    # I compute the potential energy of my halo in (Megameter/S)**2
    V_tot = np.sum(V_all)*0.5 # facteur 1/2 formule
    q = 2*K_tot/np.abs(V_tot)
    return(q)

def compute_virial_ratio(vel,pos,N_part) :
    vel_mean = np.mean(vel,axis=0)
    vel -= vel_mean
    #vel *= v_ren
    K_tot = kinetic(vel)
    pos_Dehnen = np.reshape(pos,N_part*3)
    pos_Dehnen = np.array(pos_Dehnen,dtype=np.float32)
    q = compute_q(pos_Dehnen,K_tot,N_part)
    q_approx = compute_q_approx(pos,K_tot,N_part)
    return(q,q_approx)


if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
    names_all = np.array(pd.read_csv(path+'halo_properties/names_sel.dat'))[:,0]
    if do_complete == True :
        N_haloes = len(names_all)
    prop_all = np.array(pd.read_csv(path+'halo_properties/properties_numbers_sel.dat'))
    Rvir_all = prop_all[:,0]
    Nvir_all = np.array(prop_all[:,1],dtype=int)
    cdm_all = prop_all[:,5:8]
    print('L_box =',L_box)
    q_s, cdm_cdm_vir = np.zeros((N_haloes,3)), np.zeros((N_haloes,3))
    for h in range(N_haloes) :
        halo_name = names_all[h].replace('.dat','_sphere_1_Rvir.dat')
        print(h,halo_name)
        data =  np.array(pd.read_csv(path+'data/sphere_1_Rvir/'+halo_name,sep=',', comment='#'))
        pos = data[:,0:3]
        vel = data[:,3:6]
        Rvir = Rvir_all[h]
        Nvir = Nvir_all[h]
        cdm = cdm_all[h]
        pos_frame = put_in_frame(pos,Nvir,cdm,L_box)
        cdm_cdm_vir[h] = np.mean(pos_frame,axis=0) # in Mpc/h
        q_s[h,2] = np.sqrt( (cdm_cdm_vir[h,0])**2 + (cdm_cdm_vir[h,1])**2 + (cdm_cdm_vir[h,2])**2)/Rvir
        x, y, z = pos_frame[:,0], pos_frame[:,1], pos_frame[:,2]
        q_s[h,0:2] = compute_virial_ratio(vel,pos_frame,Nvir)
    print(q_s)
    print(cdm_cdm_vir)
    if save_file == True :
        file_save = open(path+'halo_properties/q_and_s_relaxed.dat', 'w')
        file_save.write('# relaxation parameters \n')
        file_save.write('# L_box=648Mpc/h, '+cosmo+'w7, z = '+z_str+' '+mass_str+' haloes, with Brian&Norman98 selection \n')
        file_save.write('# q, q_approx, s=np.sqrt((cdm_pot - cdm_cdm_vir)**2)/Rvir  \n')
        np.savetxt(file_save,q_s)
        file_save.close()
        
        file_save_c = open(path+'halo_properties/cdm_cdm_vir.dat', 'w')
        file_save_c.write('# center of mass in the sphere of 1 Rvir (Brian&Norman98) \n')
        file_save_c.write('# L_box=648Mpc/h, '+cosmo+'w7, z = '+z_str+' '+mass_str+' haloes, with Brian&Norman98 selection \n')
        file_save_c.write('# cdm_x, cdm_y, cdm_z (Mpc/h, in the frame of the gravitational potential center)  \n')
        np.savetxt(file_save_c,cdm_cdm_vir)
        file_save_c.close()
