#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:58:41 2021

@author: guillaume
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd

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
            

def compute_ellipsoid_parameters(x,y,z,N_part_compare_ell,N_part_compare_shell,
                                 error_initial=0.001,Initial_number_iteration=10,
                                 time_duration_limit=5, frac_N_part_lost=0.1,
                                 r_min=0,r_max=1,start_ellipse=0.0001) :
    # no need to put a large number for Initial_number_iteration
    # be carefull, a low error do not necessarily says that the computation is precised
    # for a low number of particles, you can not hope to have a precise computation
    N_particles = len(x)
    simple_shell = []
    ###### Initialization
    a, b, c = 1, 1, 1 # axis length in unit of Rvir of the ellipse, it will change at each iteration
    s, q = c/a, b/a
    Sphericity, Elongation = np.array((s)),np.array((q)) # I will put all the values of sphericty and elongation here, for each iteration
    Passage = np.identity(3) # passage matrix between two frame, it will change at each iteration
    error = error_initial # at the begining, I try to get convergence with error_initial, if it does not work I will try with a higher error
    # initial inverse of passage matrix, it will contain the inverse global passage matrix between the original frame and the final frame
    Passage_general_inverse = np.linalg.inv(Passage) 
    frac_N_part_suppress = 1
    i = 0
    my_error_S, my_error_Q, my_error_max = 1, 1, 1
    time_begin = time.time() # time where I begin the while loop
    while frac_N_part_suppress > start_ellipse or my_error_max > error_initial :
    #for i in range(Initial_number_iteration) :
        #print('iteration i =',i)
        N_particles_ellipse_old = x.size
        # I put the particles position in a matrix 3*N_particles
        position = np.matrix(np.array((x,y,z)))
        # I multiply the position by the passage matrix, to go in the natural frame of the ellipsoid
        P_inv = np.linalg.inv(Passage)
        position_new = np.array((np.dot(P_inv,position)))
        x, y, z = position_new[0], position_new[1], position_new[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        r_ellipse = np.sqrt( x**2 + (y/q)**2 + (z/s)**2 )
        ind_el = np.argsort(r_ellipse)
        #print(r_ellipse[ind_el][0],r_ellipse[ind_el][-1])
        # I select only particles inside the ellipsoid, all in unit of Rvir
        ind_ellipse = np.where( r_ellipse <= r_max )[0]
        #ind_ellipse = np.where(r_ellipse < 1)[0]
        x, y, z, r_ellipse = x[ind_ellipse], y[ind_ellipse], z[ind_ellipse], r_ellipse[ind_ellipse]
        N_particles_ellipse = x.size
        #print('N_part in ellipse = ',N_particles_ellipse)
        if i > Initial_number_iteration :
            frac_N_part_suppress = 1 - N_particles_ellipse/N_particles_ellipse_old
            my_error_S = np.abs(sum(Sphericity[-Initial_number_iteration:])/(Initial_number_iteration * Sphericity[-1]) - 1)
            my_error_Q = np.abs(sum(Elongation[-Initial_number_iteration:])/(Initial_number_iteration * Elongation[-1]) - 1)
        my_error_max = np.max(np.array((my_error_S,my_error_Q)))
        #print('frac =',frac_N_part_suppress)
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        #a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
        a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r(x,y,z,1,N_particles_ellipse)
        #print('a =',a,'b =', b,'c =', c)
        #print('S =',s,'Q =', q,'T =', Triaxiality)
        # I add a new value of sphericity and Elongation
        Sphericity, Elongation = np.append(Sphericity,s), np.append(Elongation,q)
        # I put the global inverse of the passage matrix, to go from the initial frame to the final one here
        Passage_general_inverse = np.dot(np.linalg.inv(Passage),Passage_general_inverse)
        time_duration = time.time() - time_begin # time duration from the begining of the while loop
        # If the time to get convergence is too important, it means that it won't converge and so I increase the error and retry with this new error
        if time_duration > time_duration_limit :
            print('The wished convergence is not reached')
            error = error*1.1 # new error
            time_begin = time.time() # new begining time of the while loop
        # If the particles number becomes significantly lower, then I stop
        i += 1
        if N_particles_ellipse < N_part_compare_ell * frac_N_part_lost :
            frac = np.round(N_particles_ellipse/N_part_compare_ell,2) * 100
            error=0 # to say that there was a problem
            print('Be carefull, you have lost ',frac,'% of your initial particles !')
            print('The convergence is not reached for ellipsoid')
            break # stop the loop    
    simple_ell = [a,b,c,s,q,Triaxiality,error,i,N_particles_ellipse]
    Passage_ell = np.linalg.inv(Passage_general_inverse)
    time_begin = time.time() # time where I begin the while loop
    # I do a while loop, and try to find a convergence for Sphericity and Elongation values 
    # I want that the Initial_number_iteration values of Sphericity and Elongation to be very close from each other
    iteration = 0
    my_error_S, my_error_Q, my_error_max = 1, 1, 1
    #while np.abs(sum(Sphericity[-Initial_number_iteration:])/(Initial_number_iteration * Sphericity[-1]) - 1) > error or np.abs(sum(Elongation[-Initial_number_iteration:])/(Initial_number_iteration * Elongation[-1]) - 1) > error :
    while my_error_max > error_initial :
        #break
        #print('iteration =',iteration)
        #print('error =',error)
        # I put the particles position in a matrix 3*N_particles
        position = np.matrix(np.array((x,y,z)))
        # I multiply the position by the inverse passage matrix, to go in the natural frame of the ellipsoid
        position_new = np.array((np.dot(np.linalg.inv(Passage),position)))
        x,y,z = position_new[0],position_new[1],position_new[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        r_ellipse = np.sqrt(x**2 + (y/q)**2 + (z/s)**2)
        # I select particles inside the ellipsoid, all in unit of Rvir
        #ind_ellipse = np.where(r_ellipse<1)
        ind_r = np.argsort(r_ellipse)
        #print(r_ellipse[ind_r][0],r_ellipse[ind_r][-1])
        ind_ellipse = np.where( (r_min <= r_ellipse) & (r_ellipse <= r_max))[0]
        
        #ind_ellipse = np.where(r_min <= r_ellipse)[0]
        x, y, z, r_ellipse = x[ind_ellipse], y[ind_ellipse], z[ind_ellipse], r_ellipse[ind_ellipse]
        N_particles_ellipse = x.size
        #print('N_part in the ellipsoidal shell = ',N_particles_ellipse)
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        #a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
        a, b, c, s, q, Triaxiality, Passage, Inertia_tensor = triaxiality_with_r(x,y,z,1,N_particles_ellipse)
        #print('a =',a,'b =', b,'c =', c)
        #print('S =',s,'Q =', q,'T =', Triaxiality)
        # I add a new value 
        Sphericity, Elongation = np.append(Sphericity,s), np.append(Elongation,q)
        if iteration > Initial_number_iteration :
            my_error_S = np.abs(sum(Sphericity[-Initial_number_iteration:])/(Initial_number_iteration * Sphericity[-1]) - 1)
            my_error_Q = np.abs(sum(Elongation[-Initial_number_iteration:])/(Initial_number_iteration * Elongation[-1]) - 1)
        my_error_max = np.max(np.array((my_error_S,my_error_Q)))
        #print('error =',my_error_max)
        # I put the global inverse of the passage matrix, to go from the initial frame to the final one here
        Passage_general_inverse = np.dot(np.linalg.inv(Passage),Passage_general_inverse)
        time_duration = time.time() - time_begin # time duration from the begining of the while loop
        # If the time to get convergence is too important, it means that it won't converge and so I increase the error and retry with this new error
        if time_duration > time_duration_limit :
            print('The wished convergence is not reached')
            error = error*1.1 # new error
            time_begin = time.time() # new begining time of the while loop
        # If the particles number becomes significantly lower, then I stop
        if N_particles_ellipse < N_part_compare_shell * frac_N_part_lost :
            frac = 100 - np.round(N_particles_ellipse/N_part_compare_shell,2) * 100
            print('Be carefull, you have lost ',frac,'% of your initial particles !')
            print('The convergence is not reached for shell')
            error=0 # to say that there was a problem
            simple_shell = [a,b,c,s,q,Triaxiality,error,iteration,N_particles_ellipse]
            break
        
        iteration += 1
        # I put the results here
        simple_shell = [a,b,c,s,q,Triaxiality,error,iteration,N_particles_ellipse]
        Passage_shell = np.linalg.inv(Passage_general_inverse)
        
    #plt.figure(1)
    #plt.scatter(x,y,c='r',s=0.1)
    #plt.show()
    '''print('N_tot initial = ',N_particles,'N_part in ellipse = ',N_particles_ellipse)
    print('total iteration =',iteration+Initial_number_iteration,'final error =',error)
    print('a =',a,'b =', b,'c =', c)
    print('S =',s,'Q =', q,'T =', Triaxiality)'''
    return(simple_shell,Passage_shell,simple_ell,Passage_ell)


def compute_angle(v_1,v_2) :
    # v_1 dot v_2 = v_1_norm * v_2_norm * cos(theta)
    # theta =arcos( v_1 dot v_2/(v_1_norm*v_2_norm))
    scalar_product = np.dot(v_1,v_2)
    v_1_norm = np.sqrt(v_1[0]**2 + v_1[1]**2 + v_1[2]**2)
    v_2_norm = np.sqrt(v_2[0]**2 + v_2[1]**2 + v_2[2]**2)
    my_cos_theta = scalar_product/(v_1_norm*v_2_norm)
    if my_cos_theta > 1 :
        my_cos_theta -= 0.0001
    if my_cos_theta < -1 :
        my_cos_theta += 0.0001
    theta = np.arccos(my_cos_theta)
    return(theta)


def compute_change_angle(e_x_fin_shell,e_y_fin_shell,e_z_fin_shell,
                         e_x_fin_ell,e_y_fin_ell,e_z_fin_ell) :
    N_bin = len(e_x_fin_shell)
    angle_x_shell, angle_y_shell, angle_z_shell = np.zeros((N_bin)), np.zeros((N_bin)), np.zeros((N_bin))
    angle_x_ell, angle_y_ell, angle_z_ell = np.zeros((N_bin)), np.zeros((N_bin)), np.zeros((N_bin))
    for b in range(N_bin) :
        angle_x_shell[b] = compute_angle(e_x_fin_shell[-1],e_x_fin_shell[b])
        angle_y_shell[b] = compute_angle(e_y_fin_shell[-1],e_y_fin_shell[b])
        angle_z_shell[b] = compute_angle(e_z_fin_shell[-1],e_z_fin_shell[b])
        angle_x_ell[b] = compute_angle(e_x_fin_ell[-1],e_x_fin_ell[b])
        angle_y_ell[b] = compute_angle(e_y_fin_ell[-1],e_y_fin_ell[b])
        angle_z_ell[b] = compute_angle(e_z_fin_ell[-1],e_z_fin_ell[b])
        if angle_x_shell[b] > np.pi/2 :
            angle_x_shell[b] -= np.pi
        if angle_y_shell[b] > np.pi/2 :
            angle_y_shell[b] -= np.pi
        if angle_z_shell[b] > np.pi/2 :
            angle_z_shell[b] -= np.pi
        if angle_x_ell[b] > np.pi/2 :
            angle_x_ell[b] -= np.pi
        if angle_y_ell[b] > np.pi/2 :
            angle_y_ell[b] -= np.pi
        if angle_z_ell[b] > np.pi/2 :
            angle_z_ell[b] -= np.pi
    return(angle_x_shell,angle_y_shell,angle_z_shell,
           angle_x_ell,angle_y_ell,angle_z_ell)

if __name__ == '__main__':
    path_data = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/'
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/'
            #/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/properties/relaxed_halos
    path_fake = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/output_not_MF/test_halo_from_data/'
    path_prop = path+'properties/relaxed_halos/'
    #path_data = path_data+'/data/sphere_sel_1_Rvir/'
    path_data = path_data+'/data/'
    name_prop = 'prop_relaxed_b_good.dat' #'prop_relaxed.dat' # contains Rvir, Nvir, cdm_fof, cdm_dens
    name_names = 'names_relaxed_b_good.dat' #'names_relaxed.dat' # contains halo names
    prop = np.loadtxt(path_prop+name_prop)
    Rvir = prop[:,0]
    Nvir = prop[:,1]
    cdm_fof = prop[:,2:5]
    cdm_dens = prop[:,5:9]
    names = np.loadtxt(path_prop+name_names,dtype=str)
    abc = np.loadtxt(path_prop+'abc_relaxed_b_good.dat')
    print(len(names))
    my_push = 0.4
    N_halos = 10 #len(names) #150
    N_bin = 6
    probe_radius, R, res, r_s = 3, 1, 0.1, 0.1
    N_bin_creation = N_bin #15
    my_r = np.logspace(np.log10(res/r_s),np.log10(probe_radius*R/r_s),N_bin_creation)
    my_r = np.append(np.array((0)),my_r)
    my_r *= r_s
    rad = (my_r[1:] + my_r[:-1])/2
    print('full radius =',my_r)
    probe_radius, R, res, r_s = 10, 1, 0.01, 0.1
    N_bin_creation = 1000
    my_r_th = np.logspace(np.log10(res/r_s),np.log10(probe_radius*R/r_s),N_bin_creation)
    my_r_th = np.append(np.array((0)),my_r_th)
    my_r_th *= r_s
    r_0b, r_0c = my_r_th[-1], my_r_th[-1]
    b_0, c_0 = 0.5, 0.1
    s_b, s_c = (1-b_0)/np.log10(my_r_th[1]/r_0b), (1-c_0)/np.log10(my_r_th[1]/r_0c) 
    b_ax = s_b * np.log10(my_r_th[1:]/r_0b) + b_0
    c_ax = s_c * np.log10(my_r_th[1:]/r_0c) + c_0
    rad_th = (my_r_th[1:] + my_r_th[:-1])/2
    
    my_array = np.zeros((N_halos,N_bin,8),dtype=np.float32) # r_min, r_max, S, Q, T, error, N_it, N_part
    my_array[:,:,0] = my_r[0:-1]
    my_array[:,:,1] = my_r[1:]
    S_shell = np.zeros((N_halos,N_bin,11))
    Passage_array_shell = np.zeros((N_halos,N_bin,11))
    S_ell = np.zeros((N_halos,N_bin,11))
    Passage_array_ell = np.zeros((N_halos,N_bin,11))
    t0 = time.time()
    #path_data = path_fake
    #name = 'rho_abg_131_r_minus_2_0_10_r_probe_10_r_hole_0_0_Npart_100000_100100000_new_1.dat'
    for h in range(N_halos) :
        print('halo =',h)
        b_0, c_0 = abc[h,1], abc[h,2]
        #b_0, c_0 = 0.5, 0.1
        print('previous sphericity =',b_0,c_0)
        s_b, s_c = (1-b_0)/np.log10(my_r_th[1]/r_0b), (1-c_0)/np.log10(my_r_th[1]/r_0c) 
        b_ax = s_b * np.log10(my_r_th[1:]/r_0b) + b_0
        c_ax = s_c * np.log10(my_r_th[1:]/r_0c) + c_0
        #name_use = name+str(h)+'.dat'
        #name = names[h].replace('.dat','_sphere_sel_1_Rvir.dat')
        name = names[h].replace('.dat','.dat_5_Rvir_new.dat')
        print(names[h])
        #pos = np.loadtxt(path_data+name)
        data = np.loadtxt(path_data+name) # in Mpc/h
        x, y, z = data[0], data[1], data[2]
        x,y,z,bad_periodicity = periodicity(x,y,z,cdm_dens[h]/L_box)
        N_part_tot = len(x)
        pos = np.zeros((N_part_tot,3))
        pos[:,0], pos[:,1] , pos[:,2]  = x, y, z
        pos *= L_box
        pos -= cdm_dens[h]
        pos /= Rvir[h]
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        radius = np.sqrt(x**2 + y**2 + z**2)
        e_x_in, e_y_in, e_z_in = np.array((1,0,0)), np.array((0,1,0)), np.array((0,0,1))
        e_x_fin_shell, e_y_fin_shell, e_z_fin_shell = np.zeros((N_bin,3)), np.zeros((N_bin,3)), np.zeros((N_bin,3))
        e_x_fin_ell, e_y_fin_ell, e_z_fin_ell = np.zeros((N_bin,3)), np.zeros((N_bin,3)), np.zeros((N_bin,3))
        angle_x_shell, angle_y_shell, angle_z_shell = np.zeros((N_bin)), np.zeros((N_bin)), np.zeros((N_bin))
        angle_x_ell, angle_y_ell, angle_z_ell = np.zeros((N_bin)), np.zeros((N_bin)), np.zeros((N_bin))
        for b in range(N_bin) :
            #print('bin b =',b)
            r_min, r_max = my_r[b], my_r[b+1] #0.1, 0.11 # 0.84, 0.86, 0.89 # 0.94 0.95, 0.95
            S_shell[h,b,0], S_ell[h,b,0] = r_min, r_min
            Passage_array_shell[h,b,0], Passage_array_ell[h,b,0] = r_min, r_min
            S_shell[h,b,1], S_ell[h,b,1] = r_max, r_max
            Passage_array_shell[h,b,1], Passage_array_ell[h,b,1] = r_max, r_max
            print('r_min =',r_min,'r_max =',r_max)
            ind_r = np.where((r_min <= radius) & (radius <= r_max))[0]
            N_part_spherical_shell = len(ind_r)
            ind_r = np.where(radius <= r_max)[0]
            N_part_spherical_ell = len(ind_r)
            #print('N_part in spherical shell =',len(ind_r))
            x_use, y_use, z_use = x/r_max, y/r_max, z/r_max
            r_min, r_max = r_min/r_max, r_max/r_max
            #x_use, y_use, z_use = x, y, z
            S_shell[h,b,2:], P_shell, S_ell[h,b,2:], P_ell = compute_ellipsoid_parameters(x_use,y_use,z_use,N_part_compare_ell=N_part_spherical_ell,N_part_compare_shell=N_part_spherical_shell,r_min=r_min,r_max=r_max)
            #sphericity_shell[h,b,0:6] = simple_shell[0:6]
            #print('S_shell =',S_shell[h,b,5],'Q_shell =',S_shell[h,b,6])
            #sphericity_shell[h,b,6:] = simple_shell[7:]
            #Passage_shell = simple_shell[6]
            #sphericity_ell[h,b,0:6] = simple_ell[0:6]
            #print('S_ell =',S_ell[h,b,5],'Q_ell =',S_ell[h,b,6])
            #sphericity_ell[h,b,6:] = simple_ell[7:]
            #Passage_ell = simple_ell[6]
            # test passage matrix error
            e_x_fin_shell[b] = np.dot(np.linalg.inv(P_shell),e_x_in)
            e_y_fin_shell[b] = np.dot(np.linalg.inv(P_shell),e_y_in)
            e_z_fin_shell[b] = np.dot(np.linalg.inv(P_shell),e_z_in)
            
            e_x_fin_ell[b] = np.dot(np.linalg.inv(P_ell),e_x_in)
            e_y_fin_ell[b] = np.dot(np.linalg.inv(P_ell),e_y_in)
            e_z_fin_ell[b] = np.dot(np.linalg.inv(P_ell),e_z_in)
            
            Passage_array_shell[h,b,2:] = np.reshape(P_shell,(1,9))[0]
            Passage_array_ell[h,b,2:] = np.reshape(P_ell,(1,9))[0]
        
        dec = 4
        fr_shell = pd.DataFrame(np.round(S_shell[h],decimals=dec),
                                columns=['r_min','r_max','a(r_max)','b(r_max)','c(r_max)',
                                         'S_shell','Q_shell','T_shell',
                                         'error','N_it','N_shell'])
        name = names[h].replace('.dat','_shell.csv')
        fr_shell.to_csv(path_prop+'sphericity/'+name,
                        float_format='%.'+str(dec)+'f',sep=",",encoding='utf-8')
        
        fr_ell = pd.DataFrame(np.round(S_ell[h],decimals=dec),
                                columns=['r_min','r_max','a(r_max)','b(r_max)','c(r_max)',
                                         'S_ell','Q_ell','T_ell',
                                         'error','N_it','N_ell'])
        name = names[h].replace('.dat','_ell.csv')
        fr_ell.to_csv(path_prop+'sphericity/'+name,
                        float_format='%.'+str(dec)+'f',sep=",",encoding='utf-8')
        
        data = pd.read_csv(path_prop+'sphericity/'+name)
        #print(data)
        data = np.array((data))
        #print(data)
        
        fr_p_shell = pd.DataFrame(np.round(Passage_array_shell[h],decimals=dec),
                                columns=['r_min','r_max',
                                         'P11','P12','P13',
                                         'P21','P22','P23',
                                         'P31','P32','P33'])
        name = names[h].replace('.dat','_passage_shell.csv')
        fr_p_shell.to_csv(path_prop+'sphericity/'+name,
                        float_format='%.'+str(dec)+'f',sep=",",encoding='utf-8')
        
        fr_p_ell = pd.DataFrame(np.round(Passage_array_ell[h],decimals=dec),
                                columns=['r_min','r_max',
                                         'P11','P12','P13',
                                         'P21','P22','P23',
                                         'P31','P32','P33'])
        name = names[h].replace('.dat','_passage_ell.csv')
        fr_p_ell.to_csv(path_prop+'sphericity/'+name,
                        float_format='%.'+str(dec)+'f',sep=",",encoding='utf-8')
        
    print('err_pb_shell =',np.where(S_shell[:,:,8] != 0.001)[0])
    print('err_pb_ell =',np.where(S_ell[:,:,8] != 0.001)[0])
    
    #print(sphericity_ell[:,:,6])
            
    #angle_x_shell,angle_y_shell,angle_z_shell, angle_x_ell,angle_y_ell,angle_z_ell = compute_change_angle(e_x_fin_shell,e_y_fin_shell,e_z_fin_shell,
    #                                                                                                      e_x_fin_ell,e_y_fin_ell,e_z_fin_ell)
    #my_ind = 0
    #print(np.max(angle_x_shell[my_ind:]*180/np.pi), np.max(angle_y_shell[my_ind:]*180/np.pi), np.max(angle_z_shell[my_ind:]*180/np.pi))
    #print(np.max(angle_x_ell[my_ind:]*180/np.pi), np.max(angle_y_ell[my_ind:]*180/np.pi), np.max(angle_z_ell[my_ind:]*180/np.pi))
    t1 = time.time()
    print(t1-t0)
    print('S_shell =',S_shell[:,:,5])
    print('Q_shell =',S_shell[:,:,6])
    print('N_shell =',S_shell[:,:,10])
    print('N_shell_sum =',np.sum(S_shell[:,:,10],axis=1))
    print('S_ell =',S_ell[:,:,5])
    print('Q_ell =',S_ell[:,:,6])
    print('N_ell =',S_ell[:,:,10])

    rad = rad[0:N_bin]
    #rad_th = rad_th[0:N_bin]
    plt.figure(3)
    for i in range(N_halos) :
        if i == 0 :
            plt.semilogx(rad,S_shell[i,:,5],c='r',ls='-',marker='o',label='S shell')
            plt.semilogx(rad,S_shell[i,:,6],c='r',ls='--',marker='o',label='Q shell')
            plt.semilogx(rad,S_ell[i,:,5],c='b',ls='-',marker='o',label='S ell')
            plt.semilogx(rad,S_ell[i,:,6],c='b',ls='--',marker='o',label='Q ell')
        else :
            plt.semilogx(rad,S_shell[i,:,5],c='r',ls='-',marker='o')
            plt.semilogx(rad,S_shell[i,:,6],c='r',ls='--',marker='o')
            plt.semilogx(rad,S_ell[i,:,5],c='b',ls='-',marker='o')
            plt.semilogx(rad,S_ell[i,:,6],c='b',ls='--',marker='o')
    plt.semilogx(rad_th,c_ax,c='g',ls='-',label='S true')
    plt.semilogx(rad_th,b_ax,c='g',ls='--',marker='',label='Q true')
    plt.legend()
    plt.xlabel('$r_{ell}/R_{vir}$')
    plt.ylabel('S=c/a or Q=b/a')
    path_plot = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/output_not_MF/plot/plot_sphericity/'
    #plt.savefig(path_plot+'sphericity_test_100000_part_05_05.pdf')
    plt.show()
    print('full radius =',my_r)
    
    '''Passage = Passage_array_shell[0,-2]
    Passage = Passage.reshape((3,3))
    plt.figure(3)
    plt.scatter(x,y,c='r',s=0.01)
    plt.show()
    position = np.matrix(np.array((x,y,z)))
    # I multiply the position by the inverse passage matrix, to go in the natural frame of the ellipsoid
    position_new = np.array((np.dot(np.linalg.inv(Passage),position)))
    plt.figure(4)
    plt.scatter(position_new[0],position_new[1],c='b',s=0.01)
    plt.axhline(y=0,c='k',ls='--')
    
    plt.show()'''
    #np.savetxt(path_prop+'abc_relaxed_b_good.dat',sphericity)
    #np.savetxt(path_prop+'Passage_rotation_relaxed_b_good.dat',Passage_array)
    
