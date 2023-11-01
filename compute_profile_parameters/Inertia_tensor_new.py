#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:45:35 2021

@author: guillaume
"""


'''
This script computes ellipsoidal parameters of halos a, b and c
which are the three main axis length of the ellipsoid.
It also computes the passage matrice, allowing to rotate the halo, thus leading to
x <=> a, y <=> b and z <=> c.
It has been tested by generating triaxial halos with axis in any directions,
not aligned with x,y,z and it works well (typically < 1% error for N_part > 10^4).
But be carefull, I think that when I go more in the halo center region, 
I need more particles to be accurate.
I clarely need to investigate more the validity of this code at r <~ 0.1 Rvir.
'''

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd



def triaxiality_with_r_old(x,y,z,r,N_particles,mass=1) :
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
            
def compute_ellipsoid_parameters_old(x,y,z,
                                 error_initial=0.005,Initial_number_iteration=10,
                                 time_duration_limit=3, frac_N_part_lost=0.1) :
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
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # In the initialisation, I should not do that, otherwise, 
        # it will suppress good particles.
        # I should do that only in the while loop part
        #ind_ellipse = np.where(r_ellipse < 1)[0]
        #x, y, z, r_ellipse = x[ind_ellipse], y[ind_ellipse], z[ind_ellipse], r_ellipse[ind_ellipse]
        N_particles_ellipse = x.size
        #print('N_part in ellipse = ',N_particles_ellipse)
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r_old(x,y,z,r_ellipse,N_particles_ellipse)
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
        a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r_old(x,y,z,r_ellipse,N_particles_ellipse)
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
    return(a,b,c,Sphericity,Elongation,Triaxiality,Passage)
            
def compute_ellipsoid_parameters(x,y,z,a_unit=1,
                                 error_initial=0.001,Initial_number_iteration=10,
                                 CV_iteration=50, frac_N_part_lost=0.3) :
    ''' This function computes the length main axis of an ellipsoid a, b and c
    The computation method is similar to Zemp et al : https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
    x, y and z are particle positions in the frame of the center,
    The dimension of x, y and z can be anything,
    but a_unit = a/r_sphere_max = 1/r_sphere_max , 
    where r_sphere_max = max(sqrt(x**2 + y**2 + z**2))
    The dimension of the particles positions is in unit
    error_initial is the intial error that I use in order to look for convergence
    if it does not converge after CV_iteration iterations, then I increase the allowed error
    Initial_number_iteration is the number iteration that I use in order to look 
    for the axis and the Passage matrice, without removing particles.
    I do that in order to avoid removing good particles
    frac_N_part_lost is the fraction of particles that I can remove
    if I need to remove more particles, I stop the code and return 
    the actual found (not converged) values for a, b, c ...
    I tried to test the resolution limit of this code but it seems non trivial.
    It depends maybe on the total number of particles of your data set but also of how they are fluctuating.
    e.g. you can get precise results with 10**4 particles in a halo of 1 Rvir
    but with the same number of particles inside 0.01 Rvir, it is difficult 
    to get precise results'''
    # no need to put a large number for Initial_number_iteration
    # be carefull, a low error do not necessarily says that the computation is precised
    # for a low number of particles, you can not hope to have a precise computation
    N_particles = len(x)
    ##########################################################################
    # Initialization
    a, b, c = 1, 1, 1 # axis length in unit of the largest axis of the ellipsoid, it will change at each iteration
    s, q = c/a, b/a
    Sphericity, Elongation = np.array((c/a)),np.array((b/a)) # I will put all the values of sphericty and elongation here, for each iteration
    error = error_initial # at the begining, I try to get convergence with error_initial, if it does not work I will try with a higher error
    Passage = np.identity(3) # passage matrix between two frame, it will change at each iteration
    P_inv = np.linalg.inv(Passage) # inverse passage matrice between 2 frames, it will cahnge at each iteration
    Passage_general_inverse = np.linalg.inv(Passage) # general inverse matrice: it corresponds to the change between the first and the last frame
    r_ellipse = np.sqrt( x**2 + (y/q)**2 + (z/s)**2 ) 
    # I put the particles position in a matrix 3*N_particles
    pos = np.matrix(np.array((x,y,z)))
    # I initialize with a first short loop of Initial_number_iteration
    for i in range(Initial_number_iteration) :
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        a, b, c, s, q, Tri, Passage = triaxiality_with_r(x,y,z,r_ellipse,N_particles)
        # I add a new value of sphericity and Elongation
        Sphericity, Elongation = np.append(Sphericity,s), np.append(Elongation,q)
        P_inv = np.linalg.inv(Passage) # inverse passage between 2 consecutive frames
        # I multiply the position by the inverse passage matrix, 
        # in order to go in the natural frame of the ellipsoid
        pos = np.array((np.dot(P_inv,pos)))
        x, y, z = pos[0], pos[1], pos[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        r_ellipse = np.sqrt( x**2 + (y/q)**2 + (z/s)**2 ) 
        # global inverse passage matrice
        Passage_general_inverse = np.dot(P_inv,Passage_general_inverse) 
        
    #############################################################################
    # start real computation
    # I do a while loop, and try to find a convergence for both Sphericity and Elongation values 
    # I want that the Initial_number_iteration values of Sphericity and Elongation to be very close from each other
    # I also want the while loop to occur at least once, so I ask for iteration == 0
    iteration = 0
    iteration_cv = 0 # to deal with convergence problem
    while iteration == 0 or np.abs(sum(Sphericity[-Initial_number_iteration:])/(Initial_number_iteration * Sphericity[-1]) - 1) > error or np.abs(sum(Elongation[-Initial_number_iteration:])/(Initial_number_iteration * Elongation[-1]) - 1) > error :
        # I select particles inside the ellipsoid
        ind_ellipse = np.where( r_ellipse < a/a_unit )[0]
        x, y, z, r_ellipse = x[ind_ellipse], y[ind_ellipse], z[ind_ellipse], r_ellipse[ind_ellipse]
        N_particles_ellipse = x.size
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        a, b, c, s, q, Tri, Passage = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
        # I add a new value of sphericity and Elongation
        Sphericity, Elongation = np.append(Sphericity,s), np.append(Elongation,q)
        P_inv = np.linalg.inv(Passage) # inverse passage between 2 consecutive frames
        # global inverse passage matrice
        pos = np.array((np.dot(P_inv,pos)))
        x, y, z = pos[0], pos[1], pos[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        r_ellipse = np.sqrt( x**2 + (y/q)**2 + (z/s)**2 ) 
        Passage_general_inverse = np.dot(P_inv,Passage_general_inverse) 
        #############################################################################
        # test if problem of convergence
        if iteration_cv > CV_iteration : # The error wished is too small, I need to increase it
            print('The wished convergence is not reached')
            error *= 1.1 # new error
            iteration_cv = 0
        # If the particles number becomes significantly lower, then I stop
        # fraction of particles inside the ellipsoid
        frac_ell = N_particles_ellipse/N_particles 
        if frac_ell < frac_N_part_lost : # not enough particle => end the loop
            frac = 100 - np.round(frac_ell,2) * 100
            print('Be carefull, you have lost ',frac,'% of your initial particles !')
            print('The convergence is not reached')
            error=0 # to say that there was a problem
            simple = [a,b,c,c/a,b/a,Tri,
                      np.linalg.inv(Passage_general_inverse),
                      error,iteration,frac_ell]
            break
        # I put the results here
        simple = [a,b,c,c/a,b/a,Tri,
                  np.linalg.inv(Passage_general_inverse),
                  error,iteration,frac_ell]
        iteration += 1
        iteration_cv += 1
    return(simple)

def reorganise(SQT_Passage) :
    SQT = np.zeros((9))
    SQT[0:6] = SQT_Passage[0:6]
    SQT[6:] = SQT_Passage[7:]
    Passage = SQT_Passage[6]
    Passage = np.reshape(Passage,(1,9))[0]
    return(SQT,Passage)

def do_computation(path,name,add_name,unit=1,r_max=1,kind='sphere') :
    # pos can be originally in some other unit than unit
    # r_max should be in unit unit,
    # e.g. pos can be in Mpc/h, unit can Rvir in Mpc/h and r_max can be directly in Rvir
    name_use = name.replace('.dat',add_name)
    pos = np.array(pd.read_csv(path+name_use,comment='#'))
    #pos = np.loadtxt(path+name_use)
    #pos *= 10
    #r_max = 0.05 # in the same unit as pos # 0.02 (10**6), 0.06(10**5) 0.09 (10**4)
    pos /= unit # I reset in Rvir unit
    #r_max /= unit
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    my_max = np.max(r)
    ind = np.where(r < r_max)[0]
    x, y, z, r = x[ind], y[ind], z[ind], r[ind]
    my_max_2 = np.max(r)
    if my_max != my_max_2 :
        print('r_max not equal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
        print('r_max_sel =',my_max_2,'r_max_before=',my_max)
        print('r_max_th=',r_max,unit,'N_part_sel=',len(r))
    # a, b, c, S, Q, T, Passage, error, initial number of iteration
    SQT_Passage = compute_ellipsoid_parameters(x,y,z,a_unit=1/r_max)
    print(kind+': a, b, c = ',SQT_Passage[0:3])
    SQT, Passage = reorganise(SQT_Passage)
    return(SQT,Passage)


if __name__ == '__main__':
    path_fake = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/output_not_MF/test_sphericity_code/'
    path_use = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_12.5_Msun_beyond_FOF/'
    cluster = False
    if cluster == True :
        path_prop = path_use+'/halo_properties/'
        names_all = np.array(pd.read_csv(path_prop+'names_all.dat'))[:,0]
        N_halos = len(names_all) # 40 # 150
    else : # laptop
        path_prop = path_use+'/halo_properties/cluster_csv/'
        names_all = np.array(pd.read_csv(path_prop+'names_all.dat'))[:,0]
        N_halos = 2 #150
    sphere, iso_beyond_Rvir, iso = True, False, False
    ############################################################################
    rad = '1' # for the selection of the data files
    fac_rad_max = 1 # typically the maximum radius in Rvir unit
    ##########################################################################
    path_sphere = path_use+'data/sphere_'+rad+'_Rvir/' #'data/spherical_selection/'
    add_name_sphere = '_sphere_'+rad+'_Rvir.dat' #'_sphere_sel_Rvir.dat'
    path_iso_beyond = path_use+'/data/isopotential_selection/'
    add_name_iso_beyond = '_isopotential_Rvir.dat'
    path_iso = path_use+'data/iso_'+rad+'_Rvir/' #'data/isopotential_selection_not_beyond_Rvir/'
    add_name_iso = '_iso_'+rad+'_Rvir.dat' #'_iso_not_beyond_Rvir.dat'
    prop = np.array(pd.read_csv(path_prop+'properties_numbers_all.dat'))
    Rvir = prop[:,0]
    SQT_sphere, Passage_sphere = np.zeros((N_halos,9)), np.zeros((N_halos,9))
    SQT_iso_beyond, Passage_iso_beyond = np.zeros((N_halos,8)), np.zeros((N_halos,9))
    SQT_iso, Passage_iso = np.zeros((N_halos,8)), np.zeros((N_halos,9))
    t0 = time.time()
    print('N_halos =',N_halos)
    for h in range(N_halos) :
        name_halo = names_all[h]
        print('h =',h,name_halo)
        unit = Rvir[h] # typically Rvir oin Mpc/h, I can use it to set pos in Rvir unit
        print('Rvir (Mpc/h) =',unit)
        #####################################################################
        if sphere == True :
            # computation of a, b and c in a sphere
            '''path_sphere = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/output_not_MF/test_coherence_idealised_halos/variation_parameters/'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_0_2_r_hole_0_0_Npart_100000_100503_0_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_1000000_100503_0_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_10000_100503_0_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_1_0_r_hole_0_0_Npart_1000000_100503_0_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_1000000_100503_0_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_10000_100503_0_rot.dat'
            name_halo = 'rho_abg_131_r_minus_2_0_1_r_probe_10_r_hole_0_0_Npart_100000_100503_rot.dat'
            add_name_sphere = '.dat'''
            SQT_sphere[h], Passage_sphere[h] = do_computation(path_sphere,name_halo,
                                                              add_name_sphere,unit,fac_rad_max,
                                                              'sphere')
            print(SQT_sphere[h])
            
        #####################################################################
        if iso_beyond_Rvir == True :
            # computation of a, b and c in an isopotential selection
            SQT_iso_beyond[h], Passage_iso_beyond[h] = do_computation(path_iso_beyond,name_halo,
                                                                      add_name_iso_beyond,unit,'iso beyond Rvir')
        
        #####################################################################
        if iso == True :
            # computation of a, b and c in an isopotential selection
            SQT_iso[h], Passage_iso[h] = do_computation(path_iso,name_halo,
                                                        add_name_iso,unit,'iso not beyond Rvir')
    t1 = time.time()
    print('time =',t1-t0)
    if sphere == True :
        SQT_sphere = pd.DataFrame(SQT_sphere,columns=['spherical selection: a','b','c','S','Q','T','err','iteration','N_part_ell_stay_frac'])
        #SQT_sphere.to_csv(path_prop+'sphericities'+add_name_sphere,index=False)
        Passage_sphere = pd.DataFrame(Passage_sphere,columns=['spherical selection: P_11','P_12','P_13','P_21','P_22','P_23','P_31','P_32','P_33'])
        #Passage_sphere.to_csv(path_prop+'Passage_rotation'+add_name_sphere,index=False)
    if iso_beyond_Rvir == True :
        SQT_iso_beyond = pd.DataFrame(SQT_iso_beyond,columns=['isopotential selection with phi beyond and < Rvir: a','b','c','S','Q','T','err','itration'])
        #SQT_iso_beyond.to_csv(path_prop+'sphericities_isopotential_beyond_sel_all.dat',index=False)
        Passage_iso_beyond = pd.DataFrame(Passage_iso_beyond,columns=['isopotential selection with phi beyond and < Rvir: P_11','P_12','P_13','P_21','P_22','P_23','P_31','P_32','P_33'])
        #Passage_iso_beyond.to_csv(path_prop+'Passage_rotation_isopotential_beyond_sel_all.dat',index=False)
    if iso == True :
        SQT_iso = pd.DataFrame(SQT_iso,columns=['isopotential selection with phi not beyond: a','b','c','S','Q','T','err','itration'])
        #SQT_iso.to_csv(path_prop+'sphericities'+add_name_iso,index=False)
        Passage_iso = pd.DataFrame(Passage_iso,columns=['isopotential selection with phi not beyond: P_11','P_12','P_13','P_21','P_22','P_23','P_31','P_32','P_33'])
        #Passage_iso.to_csv(path_prop+'Passage_rotation'+add_name_iso,index=False)
        
    
