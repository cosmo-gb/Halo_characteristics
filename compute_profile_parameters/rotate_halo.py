#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 07:29:22 2021

@author: guillaume
"""


import numpy as np
import matplotlib.pyplot as plt
import random

def rotate(pos,theta_x,theta_y,theta_z) :
    # https://fr.wikipedia.org/wiki/Matrice_de_rotation#En_dimension_trois
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    pos_m = np.matrix(np.array((x,y,z)))
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    R_x = np.matrix([[1,0,0],[0,cos_x,-sin_x],[0,sin_x,cos_x]])
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    R_y = np.matrix([[cos_y,0,sin_y],[0,1,0],[-sin_y,0,cos_y]])
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    R_z = np.matrix([[cos_z,-sin_z,0],[sin_z,cos_z,0],[0,0,1]])
    pos_m = np.dot(R_x,np.dot(R_y,np.dot(R_z,pos_m)))
    pos_new = np.array((pos_m)).transpose()
    return(pos_new)

def apply(path,name_files,N_halos) :
    theta = np.zeros((N_halos,3))
    for h in range(N_halos) :
        name = name_files[h]#.replace('0.dat','iso.dat')
        pos = np.loadtxt(path+name)
        print(pos)
        plt.figure(1)
        plt.scatter(pos[:,0],pos[:,1],c='b',s=0.01)
        plt.show()
        theta[h] = np.random.uniform(0,np.pi/2,3) #0, np.pi/2, 0
        print(theta)
        pos = rotate(pos,theta[h,0],theta[h,1],theta[h,2])
        plt.figure(2)
        plt.scatter(pos[:,2],pos[:,1],c='r',s=0.01)
        plt.show()
        print(pos)
        print(name)
        name_save = name.replace('.dat','_rot.dat')
        print(name)
        pos = np.savetxt(path+name_save,pos)
    np.savetxt(path+'theta_rotate_small.dat',theta)
    return()

if __name__ == '__main__' :
    path = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/test_coherence_idealised_halos/variation_parameters/'
    name_files = ['rho_abg_12515_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100504_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100602_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_101010_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_1_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_131_r_minus_2_0_2_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_13505_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat']
    
    N_halos = 1
    name_files = ['rho_abg_131_r_minus_2_0_1_r_probe_1_0_r_hole_0_0_Npart_1000000_100503_0.dat']
    apply(path,name_files,N_halos)