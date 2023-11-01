#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 09:38:10 2021

@author: guillaume
"""


'''
This script computes ellipsoidal parameters of halos a, b and c
which are the three main axis length of the ellipsoid.
It is for testing the coherence of the creation of idealised halos.

'''

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd



def triaxiality_with_r(x,y,z,r,N_particles,mass=1) :
    # see Zemp et al : https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
    # mass tensor
    I_11, I_22, I_33 = np.sum((x/r)**2)/N_particles, np.sum((y/r)**2)/N_particles, np.sum((z/r)**2)/N_particles
    I_12 = I_21 = np.sum((x*y/r**2))/N_particles
    I_13 = I_31 = np.sum((x*z/r**2))/N_particles
    I_23 = I_32 = np.sum((y*z/r**2))/N_particles
    # diagonaliszation
    Inertia_tensor = np.matrix([[I_11,I_12,I_13],[I_21,I_22,I_23],[I_31,I_32,I_33]])
    eigen_values, evecs = np.linalg.eig(Inertia_tensor) # it contains the eigenvalues and the passage matrix
    #print(eigen_values)
    if eigen_values.any() < 0 :
        print('aieeeeeee!!!!')
    eigen_values *= 3
    eigen_values = np.sqrt(eigen_values)
    order = np.argsort(eigen_values) # order the eigen values, see Zemp et al: https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
    Passage = np.identity(3)
    #Passage[:,0],Passage[:,1],Passage[:,2] = np.reshape(evecs[:,order[2]],3), np.reshape(evecs[:,order[1]],3), np.reshape(evecs[:,order[0]],3)
    Passage[:,0],Passage[:,1],Passage[:,2] = evecs[order[2]], evecs[order[1]], evecs[order[0]]
    a,b,c = eigen_values[order[2]], eigen_values[order[1]], eigen_values[order[0]]
    # shape
    Sphericity, Elongation, Triaxiality = c/a, b/a, (a**2 - b**2)/(a**2 - c**2)
    return(a,b,c,Sphericity,Elongation,Triaxiality,Passage,Inertia_tensor)
            
def compute_ellipsoid_parameters(x,y,z,
                                 error_initial=0.001,Initial_number_iteration=10,
                                 time_duration_limit=3, frac_N_part_lost=0.5) :
    # no need to put a large number for Initial_number_iteration
    # be carefull, a low error do not necessarily says that the computation is precised
    # for a low number of particles, you can not hope to have a precise computation
    N_particles = len(x)
    ###### Initialization
    a, b, c = 1, 1, 1 # axis length in unit of Rvir of the ellipse, it will change at each iteration
    Sphericity, Elongation = np.array((c/a)),np.array((b/a)) # I will put all the values of sphericty and elongation here, for each iteration
    Passage = np.identity(3) # passage matrix between two frame, it will change at each iteration
    error = error_initial # at the begining, I try to get convergence with error_initial, if it does not work I will try with a higher error
    # initial inverse of passage matrix, it will contain the inverse global passage matrix between the original frame and the final frame
    Passage_general_inverse = np.linalg.inv(Passage) 
    # I initialize a first short loop of Initial_number_iteration
    for i in range(Initial_number_iteration) :
        #print('iteration i =',i)
        # I put the particles position in a matrix 3*N_particles
        position = np.matrix(np.array((x,y,z)))
        # I multiply the position by the passage matrix, to go in the natural frame of the ellipsoid
        P_inv = np.linalg.inv(Passage)
        position_new = np.array((np.dot(P_inv,position)))
        x, y, z = position_new[0], position_new[1], position_new[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        s, q = c/a, b/a
        r_ellipse = np.sqrt( (x/a)**2 + (y/b)**2 + (z/c)**2 ) * a
        # I select only particles inside the ellipsoid, all in unit of Rvir
        # Be carefull, I should not do that on the first iteration,
        # otherwise I will suppress good particles. 
        # I first need to find an approximation of the ellipsoid axis first,
        # then I can remove particles (in the while loop)
        #ind_ellipse = np.where(r_ellipse < 1)[0]
        #x, y, z, r_ellipse = x[ind_ellipse], y[ind_ellipse], z[ind_ellipse], r_ellipse[ind_ellipse]
        N_particles_ellipse = x.size
        #print('N_part in ellipse = ',N_particles_ellipse)
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
        #print('a =',a,'b =', b,'c =', c)
        #print('S =',s,'Q =', q,'T =', Triaxiality)
        # I add a new value of sphericity and Elongation
        Sphericity, Elongation = np.append(Sphericity,c/a), np.append(Elongation,b/a)
        # I put the global inverse of the passage matrix, to go from the initial frame to the final one here
        Passage_general_inverse = np.dot(np.linalg.inv(Passage),Passage_general_inverse)
        # If the particles number becomes significantly lower, then I stop
        if N_particles_ellipse < N_particles * frac_N_part_lost :
            frac = np.round(N_particles_ellipse/N_particles,2) * 100
            error=0 # to say that there was a problem
            break # stop the loop    
    simple = [a,b,c,c/a,b/a,Triaxiality,np.linalg.inv(Passage_general_inverse),error,Initial_number_iteration]
    time_begin = time.time() # time where I begin the while loop
    # I do a while loop, and try to find a convergence for Sphericity and Elongation values 
    # I want that the Initial_number_iteration values of Sphericity and Elongation to be very close from each other
    iteration = 0
    while np.abs(sum(Sphericity[-Initial_number_iteration:])/(Initial_number_iteration * Sphericity[-1]) - 1) > error or np.abs(sum(Elongation[-Initial_number_iteration:])/(Initial_number_iteration * Elongation[-1]) - 1) > error :
        #break
        #print('iteration =',iteration)
        #print('error =',error)
        # I put the particles position in a matrix 3*N_particles
        position = np.matrix(np.array((x,y,z)))
        # I multiply the position by the inverse passage matrix, to go in the natural frame of the ellipsoid
        position_new = np.array((np.dot(np.linalg.inv(Passage),position)))
        x,y,z = position_new[0],position_new[1],position_new[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        r_ellipse = np.sqrt(x**2 + (y/(b/a))**2 + (z/(c/a))**2)
        # I select particles inside the ellipsoid, all in unit of Rvir
        ind_ellipse = np.where(r_ellipse<1)
        x, y, z, r_ellipse = x[ind_ellipse], y[ind_ellipse], z[ind_ellipse], r_ellipse[ind_ellipse]
        N_particles_ellipse = x.size
        #print('N_part in ellipse = ',N_particles_ellipse)
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
        #print('a =',a,'b =', b,'c =', c)
        #print('S =',s,'Q =', q,'T =', Triaxiality)
        # I add a new value 
        Sphericity, Elongation = np.append(Sphericity,c/a), np.append(Elongation,b/a)
        # I put the global inverse of the passage matrix, to go from the initial frame to the final one here
        Passage_general_inverse = np.dot(np.linalg.inv(Passage),Passage_general_inverse)
        time_duration = time.time() - time_begin # time duration from the begining of the while loop
        # If the time to get convergence is too important, it means that it won't converge and so I increase the error and retry with this new error
        if time_duration > time_duration_limit :
            print('The wished convergence is not reached')
            error = error*1.1 # new error
            time_begin = time.time() # new begining time of the while loop
        # If the particles number becomes significantly lower, then I stop
        if N_particles_ellipse < N_particles * frac_N_part_lost :
            frac = 100 - np.round(N_particles_ellipse/N_particles,2) * 100
            print('Be carefull, you have lost ',frac,'% of your initial particles !')
            print('The convergence is not reached')
            error=0 # to say that there was a problem
            simple = [a,b,c,c/a,b/a,Triaxiality,np.linalg.inv(Passage_general_inverse),error,Initial_number_iteration]
            break
        # I put the results here
        simple = [a,b,c,c/a,b/a,Triaxiality,np.linalg.inv(Passage_general_inverse),error,Initial_number_iteration]
        iteration += 1
    '''print('N_tot initial = ',N_particles,'N_part in ellipse = ',N_particles_ellipse)
    print('total iteration =',iteration+Initial_number_iteration,'final error =',error)
    print('a =',a,'b =', b,'c =', c)
    print('S =',s,'Q =', q,'T =', Triaxiality)'''
    return(simple)

def reorganise(SQT_Passage) :
    SQT = np.zeros((8))
    SQT[0:6] = SQT_Passage[0:6]
    SQT[6:] = SQT_Passage[7:]
    Passage = SQT_Passage[6]
    Passage = np.reshape(Passage,(1,9))[0]
    return(SQT,Passage)

def do_computation(path,name,add_name,unit=1,kind='sphere') :
    name_use = name.replace('0.dat',add_name)
    print(name_use)
    #pos = np.array(pd.read_csv(path+name_use,comment='#'))
    pos = np.loadtxt(path+name_use)
    pos /= unit
    #print(pos)
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    #print(np.min(r),np.max(r))
    # a, b, c, S, Q, T, Passage, error, initial number of iteration
    SQT_Passage = compute_ellipsoid_parameters(x,y,z)
    print(kind+'a,b,c',SQT_Passage[0:3])
    SQT, Passage = reorganise(SQT_Passage)
    return(SQT,Passage)

def save_files(SQT,Passage,path_1,path_2) :
    # abc
    file = open(path_1, 'w')
    file.write('# axis length of the ellipsoids of the created halo  \n')
    file.write('# isopotential selection of 1 Rvir (in Rvir unit), rotated randomly. \n')
    file.write('# a_1,a_2,a_3, S, Q, T, error, Iteration \n')
    np.savetxt(file,SQT)
    file.close()
    # Passage
    file = open(path_2, 'w')
    file.write('# Passage matrice of the ellipsoids of the created halo  \n')
    file.write('# isopotential selection of 1 Rvir (in Rvir unit), rotated randomly. \n')
    file.write('# P_11,P_12,P_13,P_21,P_22,P_23,P_31,P_32,P_33 \n')
    np.savetxt(file,Passage)
    file.close()
    return()
    
if __name__ == '__main__':
    path_fake = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/output_not_MF/test_coherence_idealised_halos/variation_parameters/'
    #name_halo = 'rho_abg_131_r_minus_2_0_10_r_probe_10_r_hole_0_0_Npart_100000_100050030_new_1.dat'
    name_files = ['rho_abg_12515_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100504_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100602_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_101010_0.dat',
                  'rho_abg_131_r_minus_2_0_1_r_probe_1_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_131_r_minus_2_0_2_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat',
                  'rho_abg_13505_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_0.dat']
    
    add_name = 'rot.dat'
    unit = 1
    N_halos = 2
    SQT, Passage = np.zeros((N_halos,8)), np.zeros((N_halos,9))
    for h in range(N_halos) :
        name_halo = name_files[h]
        SQT[h], Passage[h] = do_computation(path_fake,name_halo,add_name,
                                            unit=unit,kind='')
    print(SQT)
    print(Passage)
    path_1 = path_fake+'abc_iso_rotated.dat'
    path_2 = path_fake+'Passage_iso_rotated.dat'
    #save_files(SQT,Passage,path_1,path_2)
    