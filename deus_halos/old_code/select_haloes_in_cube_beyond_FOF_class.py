#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:21:12 2020

@author: guillaume
"""

"""
This code looks for paticles of a halo in the full simulation box. The goal is to go beyond the Friend of Friend (FOF) algorithm
and select ourself the particles in the full box of simulation for each halo. 
It takes the center of mass (CDM), the Rvir (computed with the data from the FOF file) and the name of each halo.
It computes the boundaries of each of the 512 cubes of the simulation. Each particle belongs to one of this cube.
For each halo, with its CDM and Rvir, it assigns between 1 and 8 (in 3D) cubes, depending on if the halo is at the border of a cube.
It also checks if the cubes assigned are at the opposite side of the simulation (periodic boundary condition).
This information are computed by the two functions: 
check_halo_in_which_cube(data_properties,boundaries,factor,L_box) and find_cubes(halo_cube)
It is then given to the function: read_binary_file(name,N_particles_in_FOF,Rvir,cdm,halo_cube,factor,marge,N_cube,L_box)
"""

import struct
import numpy as np
import pandas as pd
#import unsiotools.simulations.cfalcon as falcon
import sys
import time
from random import sample
#import matplotlib.pyplot as plt

#from parameters_DEUS_fof_boxlen648_n2048_rpcdmw7 import L_box, rho_vir_Brian_Norman, mass_one_particle, rho_crit_0, Omega_matter_0, redshift, v_ren


    
    
step_log = 0.2
save_data = True
save_prop = True
cluster = True
if cluster == True : # case where I apply this code on the cluster
    if np.int(sys.argv[3]) == 0 :
        print(np.int(sys.argv[3]))
        cosmo = 'rpcdm'
    elif np.int(sys.argv[3]) == 1 :
        print(np.int(sys.argv[3]))
        cosmo = 'wcdm'
    n_parallel = np.int(sys.argv[1]) # I import a number from a terminal command line
    mass_log = np.int(sys.argv[2]) + 0.5 # mass I look for halo in log (Msun/h)
else : # case where I apply this code on my laptop
    cosmo = 'wcdm'
    mass_log = 13.5 # mass I look for halo in log (Msun/h)
    n_parallel = 128
    
    
file_param_name = 'parameters_DEUS_fof_boxlen648_n2048_'+cosmo+'w7'
file_param = __import__(file_param_name) 

L_box = file_param.L_box
rho_vir_Brian_Norman = file_param.rho_vir_Brian_Norman
mass_one_particle = file_param.mass_one_particle
rho_crit_0 = file_param.rho_crit_0
Omega_matter_0 = file_param.Omega_matter_0
redshift = file_param.redshift
v_ren = file_param.v_ren



def spherical_selection(x,y,z,R,cdm):
    # set in the center of mass frame
    #x = x - cdm[0]
    #y = y - cdm[1]
    #z = z - cdm[2]
    radius = np.sqrt( (x - cdm[0])**2 + (y - cdm[1])**2 + (z - cdm[2])**2)
    ind_sel = np.where(radius < R)[0]
    return(ind_sel)

def compute_R_delta(cdm, x, y, z, rho_delta=rho_vir_Brian_Norman):
    '''This function computes R_delta, with a density threshold delta,
    cdm is the halo center (for e.g. the highest density point)
    x,y,z are the particles positions in Mpc/h
    rho_delta is the density threshold in mass_one_particle/(Mpc/h)**3
    by default, it is rho_vir_Brian_Norman 
    It returns R_delta in Mpc/h, N_delta the particle number in a sphere R_delta
    error_0 is non zero if the density was too large
    error_fof is nonzero if the density threshold was too small
    the radius_sorted in Mpc/h, the radius sorted,
    the indice ind allowing to sort the radius in Mpc/h
    the mean density sorted (as radius) rho in mass_one_particle/(Mpc/h)**3
    It also computes R200m, R200c, R500c, R1000c, R2000c
    see https://iopscience.iop.org/article/10.1088/0004-637X/792/1/25/pdf
    '''
    error_0, error_fof = 0, 0
    N_part_tot = len(x)
    # I sort the radii and compute the mean density
    radius = np.sqrt((x-cdm[0])**2 + (y-cdm[1])**2 + (z-cdm[2])**2)                                                                                                                                                                 
    ind = np.argsort(radius)                                                                                                                                 
    r_sorted = radius[ind]
    N_part_array = np.linspace(1,N_part_tot,N_part_tot)
    rho = N_part_array * 3/(4*np.pi*(r_sorted)**3)
    # I take the last indice rho > rho_lim
    ind_rho = np.where(rho > rho_delta)[0]
    error_0 = 0
    error_fof = 0
    N_delta = len(ind_rho)
    if N_delta > 0 and N_delta < N_part_tot :
        R_delta = (r_sorted[N_delta] + r_sorted[N_delta-1])/2
    elif N_delta == 0 :
        # the threshold is too large and there is no particles
        # with higher density than that threshold
        R_delta = 0
        error_0 += 1
        print('''the density threshold is too large, \n 
              you have no particles inside this threshold''')
    elif N_delta == N_part_tot :
        # The threshold is too small and all the particles given 
        # by the FOF algorithm have a higher density than this threshold. 
        # Thus, some particles which are not given by the FOF algorithm 
        # could enter in the computation of R_delta.
        # I return an approximation of R_delta
        R_delta = (3 * N_part_tot/(4 * np.pi * rho_delta))**(1/3)
        error_fof += 1
        print('''the density threshold is too small, you could have particles \n
              which are not in the input and which contribute to the computation of R_delta''')
    else:
        R_delta = -1 # Big problem !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print('N_delta should be in the interval [0,N_part_tot] and it is not the case')
    # compute also M_200_m, M_200_c, M_500_c, M_1000_c, M_2000_c
    # https://iopscience.iop.org/article/10.1088/0004-637X/792/1/25/pdf
    rho_200_m = 200 * rho_crit_0 * Omega_matter_0 * (1 + redshift)**3
    N_200_m = len(np.where(rho > rho_200_m)[0])
    if N_200_m > 0 and N_200_m < N_part_tot :
        R_200_m = (r_sorted[N_200_m] + r_sorted[N_200_m-1])/2
    elif N_200_m == 0 :
        R_200_m = 0
    elif N_200_m == N_part_tot :
        R_200_m = (3 * N_part_tot/(4 * np.pi * rho_200_m))**(1/3)
    else :
        R_200_m = -1    
    ################################################################# 200_c
    Omega_c = 1-Omega_matter_0 + Omega_matter_0 * (1 + redshift)**3
    rho_200_c = 200 * rho_crit_0 * Omega_c
    N_200_c = len(np.where(rho > rho_200_c)[0])
    if N_200_c > 0 and N_200_c < N_part_tot :
        R_200_c = (r_sorted[N_200_c] + r_sorted[N_200_c-1])/2
    elif N_200_c == 0 :
        R_200_c = 0
    elif N_200_c == N_part_tot :
        R_200_c = (3 * N_part_tot/(4 * np.pi * rho_200_c))**(1/3)
    else :
        R_200_c = -1
    ############################################################### 500_c
    rho_500_c = 500 * rho_crit_0 * Omega_c
    N_500_c = len(np.where(rho > rho_500_c)[0])
    if N_500_c > 0 and N_500_c < N_part_tot :
        R_500_c = (r_sorted[N_500_c] + r_sorted[N_500_c-1])/2
    elif N_500_c == 0 :
        R_500_c = 0
    elif N_500_c == N_part_tot :
        R_500_c = (3 * N_part_tot/(4 * np.pi * rho_500_c))**(1/3)
    else :
        R_500_c = -1
    ############################################################### 1000_c
    rho_1000_c = 1000 * rho_crit_0 * Omega_c
    N_1000_c = len(np.where(rho > rho_1000_c)[0])
    if N_1000_c > 0 and N_1000_c < N_part_tot :
        R_1000_c = (r_sorted[N_1000_c] + r_sorted[N_1000_c-1])/2
    elif N_1000_c == 0 :
        R_1000_c = 0
    elif N_1000_c == N_part_tot :
        R_1000_c = (3 * N_part_tot/(4 * np.pi * rho_1000_c))**(1/3)
    else :
        R_1000_c = -1
    ############################################################### 2000_c
    rho_2000_c = 2000 * rho_crit_0 * Omega_c
    N_2000_c = len(np.where(rho > rho_2000_c)[0])
    if N_2000_c > 0 and N_2000_c < N_part_tot :
        R_2000_c = (r_sorted[N_2000_c] + r_sorted[N_2000_c-1])/2
    elif N_2000_c == 0 :
        R_2000_c = 0
    elif N_2000_c == N_part_tot :
        R_2000_c = (3 * N_part_tot/(4 * np.pi * rho_2000_c))**(1/3)
    else :
        R_2000_c = -1
    ###########################################################################
    output_R_and_M = np.array((R_delta,N_delta,R_200_m,N_200_m,
                               R_200_c,N_200_c,R_500_c,N_500_c,
                               R_1000_c,N_1000_c,R_2000_c,N_2000_c))
    output_number = np.array((R_delta,N_delta,error_0,error_fof))
    output_array = np.array((r_sorted,ind,radius,rho))
    return(output_number, output_array, output_R_and_M)

def check_halo_in_which_cube(boundaries,cdm_i,Rlim,halo_cube,i,L_box):
    ''' This function finds in which cube of the full simulation box are 
    the halo centers of all haloes in the direction i (x <=> i=1, y <=> i=2, z <=> i=3). 
    It also finds if those centers are close (according to Rlim) of the boundary 
    of the cubes. It returns halo_cube, described below.
    Be carefull, Rlim should be smaller than L_box/N_cubes_1D
    boundaries is a 1D np.array containing the bin_edges (in Mpc/h) of the full box cut in N_cubes small cubes.
    As the simulation is a cube, it is sufficient to have it in 1D, and to re-use it in all directions i
    cdm_i are all the halo centers in the direction i (in Mpc/h).
    I want to look for particles inside a cube centered on the halo center and of size 2*Rlim.
    Rlim is a np.array containing N_haloes distances (in Mpc/h).
    halo_cube is a np.array of N_haloes lines and 7 columns, made of integers.
    At the begining, it contains only zeros, then it is filled on each direction i
    The first column contains the halo numero h between [0,N_haloes-1]
    The second to fourth columns (the c_i's) fully characterize the cube which 
    contains the halo center, one for each direction i. The c_i are between [0,N_cubes_1D-1],
    where N_cubes_1D=(N_cubes)**(1/3) (N_cubes_1S is the number of cubes in 1D).
    The fifth to seventh columns (the i_n's) characterizes if the halo center is 
    closed of the boundary of its cube, one for each direction i.
    The i_n's are between [-1,1], and
    If i_n = 0, it means that cdm_i = is far from the boundary of its cube on the i direction
    If i_n = 1, it means that cdm_i + Rlim > upper boundary of its cube on the i direction
    If i_n = -1, it means that cdm_i - Rlim < lower boundary of its cube on the i direction
    '''
    # First I check if the center is inside the simulation box [0,L_box], 
    # and I set it if it is not the case (the simulation must used periodic condition)
    ind_i_plus = np.where(cdm_i > L_box)[0]
    cdm_i[ind_i_plus] = cdm_i[ind_i_plus] - L_box
    ind_i_minus = np.where(cdm_i < 0)[0]
    cdm_i[ind_i_minus] = cdm_i[ind_i_minus] + L_box
    N_cubes_1D = len(boundaries) - 1
    for c_i in range(N_cubes_1D): # loop on the cubes inside the full box in the i direction
        bound_min = boundaries[c_i] # boundaries of the cube c_i in the i direction
        bound_max = boundaries[c_i+1]
        ind_cdm_i_sup = np.where(cdm_i > bound_min)[0]
        ind_cdm_i_inf = np.where(cdm_i[ind_cdm_i_sup] < bound_max)[0]
        ind_i = ind_cdm_i_sup[ind_cdm_i_inf] # contains only haloes inside the cube c_i
        cdm_i_used = cdm_i[ind_i] 
        Rlim_used = Rlim[ind_i] 
        N_c_i = len(ind_i) 
        i_n_used = np.zeros((N_c_i),dtype=int)
        ind_plus = np.where(cdm_i_used  - Rlim_used <= bound_min)[0]
        i_n_used[ind_plus] = -1 # lower boundary effect possible
        ind_minus = np.where(cdm_i_used + Rlim_used >= bound_max)[0]
        i_n_used[ind_minus] = 1 # upper boundary effect possible
        c_i_array = np.ones((N_c_i),dtype=int) * c_i
        halo_cube[ind_i,0], halo_cube[ind_i,i], halo_cube[ind_i,i+3] = ind_i, c_i_array, i_n_used
    return(halo_cube)

def find_cubes(halo_cube,N_cubes_1D=8):
    '''This function takes as an argument what returns the function: 
    check_halo_in_which_cube(data_properties,boundaries,factor,L_box).
    It finds explicitly all the cubes (between 1 and 8) that are assigned to each halo 
    and if these cubes are at the opposite of the full simulation box.
    It returns halo, a list, containing (1 + N_cubes + 1) lists .
    The first list contains the halo numero (1 integer between [0,N_haloes-1]).
    The last list contains n, which is an integer between [0,3] characterizing 
    the number of cubes = 2**n I need to explore, in order to look for particles.
    Thus, they are 2**n lists in between, containg each 6 integers. 
    The 3 first are the c_i's', which characterized the cube I need to look for,
    they are between [0,7]. The 3 lasts are the i_n, which characterized if the cube 
    is at the opposite to the simulation box compared to the cube containg the halo center.
    They are between [-1,1] or = 137. i_n=137 means that in the direction i, 
    I do not change of cube compared to the halo center's cube. In the other cases,
    so between [-1,1], I do change of cube in the direction i. i_n=0 means that
    the cube changed and the original cube touchs each other in the simulation box.
    i_n=1 means that the halo center cdm_i is close to 1 and the cube position coordinates 
    (x's, y's and z's) are close to 0. i_n=-1 means the opposite. 
    '''
     # n will characterize the number of cubes where I need to look for particles
    n = 0 # number of direction I need to shift, to look for in another cube
    # the number 137 means that nothing needs to be done ine the i direction, 
    N_fun = 137 # N_fun could be anything not between [-1,1]
    h = halo_cube[0] # halo numero
    # cube numero where the halo center cdm_i is in the i direction
    c_x, c_y, c_z = halo_cube[1], halo_cube[2], halo_cube[3] 
    # need to look for another cube in the i direction (i_n=-1 or 1) or not (i_n=0)
    x_n, y_n, z_n = halo_cube[4], halo_cube[5], halo_cube[6] 
    # I begin by considering the cube of the halo center, initialization
    # if all the i_n=0, I just add [n] to that list and return it
    halo = [[h],[c_x, c_y, c_z, N_fun, N_fun, N_fun]] # no shift case, 1 cube
    # cube numero of the cube neighbour in the i direction
    c_x_new, c_y_new, c_z_new = (c_x + x_n)%N_cubes_1D, (c_y + y_n)%N_cubes_1D, (c_z + z_n)%N_cubes_1D
    # The i_n_new's won't be used in the direction where i_n=0, and I will use N_fun in that direction
    # if used, it is =0 if the cube containing the halo center is not close to the boundary
    # of the full simulation box. Otherwise, it is =-1 or =1.
    x_n_new, y_n_new, z_n_new = (c_x + x_n)//N_cubes_1D, (c_y + y_n)//N_cubes_1D, (c_z + z_n)//N_cubes_1D 
    # potentially 7 more cubes than the one of the halo center, 1 cube by if
    if x_n != 0 : # at least 2 cubes (shift on x), but maybe more
        halo = halo + [[c_x_new, c_y, c_z, x_n_new, N_fun, N_fun]]
        n += 1 # I need to shift in the x direction
    if y_n != 0 : # at least 2 cubes (shift on y), but maybe more
        halo = halo + [[c_x, c_y_new, c_z, N_fun, y_n_new, N_fun]]
        n += 1 # I need to shift in the y direction
    if z_n != 0 : # at least 2 cubes (shift on z), but maybe more
        halo = halo + [[c_x, c_y, c_z_new, N_fun, N_fun, z_n_new]]
        n += 1 # I need to shift in the z direction
    # if I need more than 1 cube, I also need to consider the shift in both directions
    if x_n != 0 and y_n != 0: # at least 4 cubes (shift on x and y), but maybe more
        halo = halo + [[c_x_new, c_y_new, c_z, x_n_new, y_n_new, N_fun]]
    if x_n != 0 and z_n != 0: # at least 4 cubes (shift on x and z), but maybe more
        halo = halo + [[c_x_new, c_y, c_z_new, x_n_new, N_fun, z_n_new]]
    if y_n != 0 and z_n != 0: # at least 4 cubes (shift on y and z), but maybe more
        halo = halo + [[c_x, c_y_new, c_z_new, N_fun, y_n_new, z_n_new]]
    if x_n != 0 and y_n != 0 and z_n != 0: # 8 cubes (maximum) (shift on x, y and z)
        halo = halo + [[c_x_new, c_y_new, c_z_new, x_n_new, y_n_new, z_n_new]]
    
    halo = halo + [[n]]
    return(halo)
    
def select_particles_in_cube(x,y,z,Rlim,cdm):
    '''This function selects particles in a cube centered on cdm and of half size Rlim.
    It takes the particle positions x,y,z (in Mpc/h), velocities vx,vy,vz, and identities ide.
    It selects those which are in the cube I want (defined with cdm and Rlim).
    It returns the positions, velocities and identities of those particles, with their number N_part.
    Be carefull, it does not check the periodicity !!!
    '''   
    ind_x = np.where(np.abs(x - cdm[0]) < Rlim)[0] # particles with good x's
    y_new_x, z_new_x = y[ind_x], z[ind_x]     
    ind_x_y = np.where(np.abs(y_new_x - cdm[1]) < Rlim)[0] # particles with good x's and y's
    z_new_x_y = z_new_x[ind_x_y] 
    ind_x_y_z = np.where(np.abs(z_new_x_y - cdm[2]) < Rlim)[0] # particles with good x's, y's and z's
    ind = ind_x[ind_x_y][ind_x_y_z] # particles with good x's, y's and z's
    N_part = len(ind) # particles number which are in the cube that I want to select
    x, y, z = x[ind], y[ind], z[ind]
    return(x, y, z, ind, N_part)
       
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
        
def read_binary_file(path_cube, name_file, halo_cube, cdm, Rlim, N_marge, N_cubes, L_box):
    '''This function looks for particles close to a halo beyond the FOF algorithm.
    It takes all particles in the simulation inside a cube centered on cdm and of size 2*Rlim.
    path_cube indicates where is the cube file of DEUS in the computer
    name_file indicates the name of the cube file, e.g., fof_boxlen648_n2048_lcdmw7_cube
    halo_cube contains the identity(ies) of the cube(s) containing such particles.
    It is the output of the function find_cubes.
    cdm is the halo center, in Mpc/h.
    Rlim is thought as R_delta*factor, in Mpc/h.
    N_marge is the number of particles of the initialized arrays pos, vel and ide.
    It should be larger than the final number of particles called N_tot inside 
    the area we are looking for. It is thought as N_marge=N_particle_in_FOF*marge.
    N_cubes is the total number of cubes in the full simulation (usually it is 512).
    L_box is the size of the full simulation box, in Mpc/h.
    It returns an array of 7*N_part_tot, containing the positions (in Mpc/h),
    velocities (in km/s) and identities  of all the particles inside a cube 
    of size 2*Rlim centered on the halo center.
    '''
    print('read_binary_file')
    # check in which cube I effectively select particles at the end
    found_part = np.zeros((8),dtype=int) # the 0 will be replaced by 1 if there are indeed particles in the cube c
    N_zeros_x, N_zeros_y, N_zeros_z = 0, 0, 0 # in order to check if I do not add zeroes in my output
    # I choose positions and velocities to be np.float64 in order to avoid duplicates on positions
    # But in http://www.deus-consortium.org/a-propos/cosmological-models/post-processing/, 
    # they say that it is simple precision, meaning np.float32
    # I choose identities to be np.int64, according to deus-consortium.
    my_type_pos, my_type_vel, my_type_id = np.float64, np.float64, np.int64 # data type 
    # I will set the positions, velocities and identities of the particles 
    # I am interested in, in the following 3 arrays called pos, vel and ide. 
    # N_marge must be larger than the number of particles I want to select 
    # (I do not know this number à priori, so I take a marge)
    pos = np.zeros((N_marge,3), dtype = my_type_pos)
    vel = np.zeros((N_marge,3), dtype = my_type_vel)
    ide = np.zeros((N_marge), dtype = my_type_id)
    N_tot = 0 # total particle number that I want to select, initialized at 0
    # number of different cubes I need to explore, in order to look for particles
    N_cube_loop = np.int(2**halo_cube[-1][0]) # N_cube_loop is an integer between [1,8]
    # First I set the cdm in the 3 directions between [0,L_box]
    # because the output of DEUS are also between [0,L_box]
    # ind_cdm_plus contains the information of the upper boundary problem
    ind_cdm_plus = np.where(cdm > L_box)[0] 
    if len(ind_cdm_plus) != 0 : # at least 1 upper boundary problem
        cdm[ind_cdm_plus] = cdm[ind_cdm_plus] - L_box
    # ind_cdm_moins contains the information of the lower boundary problem
    ind_cdm_moins = np.where(cdm < 0)[0]
    if len(ind_cdm_moins) != 0 : # at least 1 lower boundary problem
        cdm[ind_cdm_moins] = cdm[ind_cdm_moins] + L_box
    # loop on the different cubes I want to explore, in order to look for particles
    for c in range(N_cube_loop): 
        # here we study the case of the cube c, characterized by the c_i's
        # I select which c_i's I will use and the corresponding i_n's, in my_cube
        # which then contains 6 elements: 3 c_i's and 3 i_n's
        my_cube = halo_cube[c+1]
        c_i = np.array((my_cube[0], my_cube[1], my_cube[2])) # c_x, c_y and c_z integers between [0,N_cube_1D-1]
        i_n = np.array((my_cube[3], my_cube[4], my_cube[5])) # x_n, y_n and z_n integers between [-1,1] or 137
        N_cube_1D = np.int(np.cbrt(N_cubes)) # cube numbers in 1 direction in the simulation box
        # numero of the file corresponding to 1 cube c_x, c_y, c_z, integer between [0,N_cubes-1]
        num = np.int(c_i[0]*N_cube_1D**2 + c_i[1]*N_cube_1D + c_i[2]) 
        # I open the binary file numero num, it corresponds to 1 of the N_cubes of the simulation
        file = open(path_cube+name_file+'_'+str('{:d}'.format(num).zfill(5))+"", "rb")                                                                                      
        struct.unpack('<i', file.read(4))  # number of bytes for the total number of particles
        N_part_tot = struct.unpack('<i', file.read(4))[0] # total number of particles in the cube c   
        struct.unpack('<i', file.read(4)) # number of bytes of procid, the identity of the binary file
        struct.unpack('<i', file.read(4))[0] # procid, the identity of the binary file
        struct.unpack('<iii', file.read(4*3)) # number of bytes of the boundaries for the position
        struct.unpack('<'+str(6)+'f', file.read(4*6)) # boundaries for the position in box=1 unit
        struct.unpack('<ii', file.read(8))  # number of bytes for the position
        # positions of all particles in the file, it lies between [0,1] (box unit)
        pos_cube = struct.unpack('<'+str(N_part_tot*3)+'f', file.read(4*3*N_part_tot)) 
        struct.unpack('<ii', file.read(8)) # number of bytes for the velocities
        # velocities of all particles in the file, in simu unit
        vel_cube = struct.unpack('<'+str(N_part_tot*3)+'f', file.read(4*3*N_part_tot)) 
        struct.unpack('<ii', file.read(8)) # number of bytes for the identities
        # identities of all particles in the file, integers between[1,N_tot_simu]
        ide_cube = struct.unpack('<'+str(N_part_tot)+'q', file.read(8*N_part_tot)) 
        pos_cube = np.array(pos_cube) * L_box # I set the positions in Mpc/h
        pos_cube = np.reshape(pos_cube,(N_part_tot,3)) # I reshape (N_part_tot lines and 3 columns)
        x, y, z = pos_cube[:,0], pos_cube[:,1], pos_cube[:,2] 
        vel_cube = np.array(vel_cube)
        vel_cube = np.reshape(vel_cube,(N_part_tot,3)) 
        ide_cube = np.array(ide_cube,dtype=my_type_id) 
        # I look for zeroes which are really in the simulation (it is a possible position)
        ind_x_zeros, ind_y_zeros, ind_z_zeros = np.where(x == 0)[0], np.where(y == 0)[0], np.where(z == 0)[0] 
        N_zeros_x, N_zeros_y, N_zeros_z = N_zeros_x + len(ind_x_zeros), N_zeros_y + len(ind_y_zeros), N_zeros_z + len(ind_z_zeros)
        N_part_new = 0
        # I check the potential periodicity problem of that cube
        # if i_n =-1 or +1 it means that the particles in that cube are in a cube which
        # is at the opposite on the direction i of the cube containing the halo center
        # and so I move the corresponding cdm for that cube c
        ind_new_cdm_plus = np.where(i_n == 1)[0]    
        ind_new_cdm_moins = np.where(i_n == -1)[0]
        cdm[ind_new_cdm_moins] = cdm[ind_new_cdm_moins] + L_box 
        cdm[ind_new_cdm_plus] = cdm[ind_new_cdm_plus] - L_box
        # I select the particles in the cube c
        x_new, y_new, z_new, ind_new, N_part_new = select_particles_in_cube(x, y, z, Rlim, cdm)
        pos_new = np.array((x_new, y_new, z_new)).transpose()
        vel_new, ide_new = vel_cube[ind_new,:], ide_cube[ind_new]
        if N_part_new > 0 :
            # If N_part_new > 0, it means that I indeed found particles inside the cube c
            # which is expected, otherwise, it is very likely that there is a problem.
            # either the cube is not the right where I need to look for.
            # But it is also possible that it is a limit issue, meaning that 
            # the volume where I need to look for inside the cube c is so small 
            # that there are no particles in it. Again, it is very unlikely.
            found_part[c] = 1
        N_beg, N_end = N_tot, N_tot + N_part_new
        if N_end > N_marge :
            print('The number of particles found N_tot is greater than N_tot > ',N_end - 1,'. \n '
                  'And so it is superior to the number of particles allowed N_marge = ',N_marge,'. \n',
                  'You should increase N_marge.')
            break
        pos[N_beg:N_end,:], vel[N_beg:N_end,:], ide[N_beg:N_end] = pos_new, vel_new, ide_new
        cdm[ind_new_cdm_moins] = cdm[ind_new_cdm_moins] - L_box # I replace the cdm
        cdm[ind_new_cdm_plus] = cdm[ind_new_cdm_plus] + L_box
        N_tot = N_tot + N_part_new # new number of particles that I am looking for
    # end loop cube
    pos, vel, ide = pos[0:N_tot,:]/L_box, vel[0:N_tot,:], ide[0:N_tot]
    x_new, y_new, z_new = pos[:,0], pos[:,1], pos[:,2]
    x_new, y_new, z_new, bad_periodicity = periodicity(x_new, y_new, z_new, cdm/L_box)
    x_new, y_new, z_new = x_new*L_box, y_new*L_box, z_new*L_box
    out_num, out_array, out_R_and_M = compute_R_delta(cdm,x_new,y_new,z_new)   
    R_delta, N_delta = out_num[0], out_num[1]
    vx_new, vy_new, vz_new = vel[:,0], vel[:,1], vel[:,2]
    halo_final = np.array((x_new/L_box, y_new/L_box, z_new/L_box, vx_new, vy_new, vz_new, ide))
    ind_sphere_delta = spherical_selection(x_new,y_new,z_new,R_delta,cdm)
    halo_sphere_delta = np.array((x_new[ind_sphere_delta],y_new[ind_sphere_delta],z_new[ind_sphere_delta],
                                  vx_new[ind_sphere_delta]*v_ren, vy_new[ind_sphere_delta]*v_ren, vz_new[ind_sphere_delta]*v_ren, 
                                  ide[ind_sphere_delta]))
    ind_pb_x, ind_pb_y, ind_pb_z = np.where(x_new == 0)[0], np.where(y_new == 0)[0], np.where(z_new == 0)[0]
    halo_final_pb = 0 # check if there are too many zeros positions in the particles
    if len(ind_pb_x) > N_zeros_x or len(ind_pb_y) > N_zeros_y or len(ind_pb_z) > N_zeros_z :
        halo_final_pb = 1 # there are too many zeros positions in the particles
    ind_found_part = np.where(found_part != 0)[0] # characterized in which cube I found particles 
    
    return(halo_final, halo_sphere_delta, ind_found_part, N_cube_loop, halo_final_pb, R_delta, N_delta, out_R_and_M)
       
def beyond_FOF_func(cdm, Rvir, path_cube, name_file, f, path_out, 
                    Mass_halo_min, Mass_halo_max, L_box=L_box,N_cubes=512,factor=5,marge=100,save_data=True):
    Rlim = factor * Rvir # Rlim used (Mpc/h) (half size of the cube where I want to take particles)
    N_loop = np.int(np.cbrt(N_cubes)) # number of cubes **(1/3) = N_cubes_1D
    boundaries = np.linspace(0,L_box,N_loop+1,dtype=float) # boundaries of the cubes in the simulation in 1D
    N_haloes = len(Rlim) # number of haloes
    # halo_cube contains : halo numero, cubes c1,c2,c3 numeros, boundaries x_n,y_n,z_n, 
    # see the function check_halo_in_which_cube for details
    halo_cube = np.zeros((N_haloes,7),dtype=int) 
    halo_cube = check_halo_in_which_cube(boundaries,cdm[0],Rlim,halo_cube,1,L_box)
    halo_cube = check_halo_in_which_cube(boundaries,cdm[1],Rlim,halo_cube,2,L_box)
    halo_cube = check_halo_in_which_cube(boundaries,cdm[2],Rlim,halo_cube,3,L_box)
    my_halo_cube = halo_cube[f]
    my_cubes = find_cubes(my_halo_cube) 
    print(my_cubes)
    cdm_f = np.array((cdm[0,f],cdm[1,f],cdm[2,f])) # halo center (Mpc/h)
    Rlim_f = Rlim[f]
    N_marge = np.int(prop[f,0]) * marge
    halo_final_all = read_binary_file(path_cube, name_file, my_cubes, cdm_f, Rlim_f, N_marge, N_cubes, L_box)
    halo_final, halo_sphere_delta, halo_final_pb = halo_final_all[0], halo_final_all[1], halo_final_all[4] # data and problem
    R_delta, N_delta = halo_final_all[5], halo_final_all[6] # Rvir and Nvir
    out_R_and_M = halo_final_all[7]
    name_f = names[f]
    name_f_save = name_f.replace('.dat','')
    mass_ok = False
    if Mass_halo_min < N_delta < Mass_halo_max : # the halo has the good mass
        print(name_f)
        mass_ok = True
        halo_final = halo_final.transpose() #reshape((N_part_tot,7))
        halo_sphere_delta = halo_sphere_delta.transpose()
        #np.savetxt(path_out+'data/'+name_f+'_'+str(factor)+'_Rvir_new.dat',halo_final)
        if save_data == True :
            halo_final_csv = pd.DataFrame(halo_final,columns=['x (DEUS unit)','y (DEUS unit)','z (DEUS unit)',
                                                              'v_x (DEUS unit)','v_y (DEUS unit)','v_z (DEUS unit)','ide'])
            halo_final_csv.to_csv(path_out+'data/'+name_f_save+'_'+str(factor)+'_Rvir.dat',index=False)
            halo_sphere_delta_csv = pd.DataFrame(halo_sphere_delta,columns=['x (Mpc/h)','y','z',
                                                                            'v_x (km/s in box frame not in halo s)','v_y','v_z','ide'])
            halo_sphere_delta_csv.to_csv(path_out+'data/'+name_f_save+'_sphere_1_Rvir.dat',index=False)
    if halo_final_pb != 0 :
        print(name_f)
        print('problem: too many zeros in the particle positions')
        N_zeroes_saved = pd.DataFrame(np.array([name_f,halo_final_pb]),columns=['halo name','number of zeroes in the final data'])
        N_zeroes_saved.to_csv(path_out+'Problem/zeroes_'+name_f_save+'.dat',index=False)
    ind_found_part, N_cube_loop = halo_final_all[2], halo_final_all[3]     
    if N_cube_loop != ind_found_part.size:   
        print(name_f)
        print('problem: the number of cubes where I effectively found particles is not equal \n'
              'to the number of cubes I looked for particles. It is not necessarly a mistake, it is just unlikely.')
        print(N_cube_loop)
        print(ind_found_part)
        print('test')
        cube_pb = np.array((f,N_cube_loop,ind_found_part.size)).reshape((1,3))
        print(cube_pb)
        cubes_pb_saved = pd.DataFrame(cube_pb,columns=['f (halo candidate from FOF numero)','Number of cubes I need to look for particles','Number of cubes where I found particles'])
        cubes_pb_saved.to_csv(path_out+'Problem/cubes_'+name_f_save+'.dat',index=False)
    return(halo_final,R_delta,N_delta,mass_ok,out_R_and_M)

def indice_cluster(n_parallel,N_haloes,N_cores):
    # if I want to apply to the cluster
    #N_haloes = 2876 # Number of haloes
    #n_use = N_haloes // N_cut
    multiple = np.int(N_haloes//(N_cores-1))
    print('N_cores =',N_cores)
    print('N_haloes =',N_haloes)
    print('multiple =',multiple)
    #multiple = 10
    if n_parallel < N_cores :
        f_begin = (n_parallel - 1) * multiple #n_use #np.int(multiple*(n-1)) # begining and end of my loop
        f_end = n_parallel * multiple #n_use #np.int(multiple*n)
    else :
        f_begin = (n_parallel - 1) * multiple #n_use #multiple*N_cores
        f_end = N_haloes
    print('f_begin,f_end =',f_begin,f_end)
    return(f_begin,f_end)


# path for load properties that I get from FOF
path_prop_fof = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+str(mass_log)+'_Msun_FOF/halo_properties/'
# path for load binary data
path_cube = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/snapshot/'  # path where I look for the binary data
# path for save the data
path_out = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+str(mass_log)+'_Msun_beyond_FOF/'

##################################################################################
# for the case mass = 12.5, I have many haloes to look for
# an idea would have been to previously select randomly a fraction of these haloes found by the FOF algorithm
prop_name = 'properties_numbers_all.dat'
#prop_name = 'properties_numbers_1.dat'
prop = pd.read_csv(path_prop_fof+prop_name)
prop = np.array(prop)
name_name = 'names_all.dat'
#name_name = 'haloes_usefull_1.dat'
names = pd.read_csv(path_prop_fof+name_name)
names = np.array(names)[:,0]
N_haloes = len(names)
N_cores = 128
f_begin, f_end = indice_cluster(n_parallel,N_haloes,N_cores)
#f_begin, f_end = 0, 2 #64, 65
N_ok_threshold = 4
name_file = 'fof_boxlen648_n2048_'+cosmo+'w7_cube' # name of file 
Mass_min = (10**mass_log)/mass_one_particle # virial mass minimum that I want for my halo in Msun/h
Mass_max = (10**(mass_log + step_log))/mass_one_particle # virial mass maximum that I want for my halo in Msun/h         
#R_delta_all, N_delta_all = np.zeros((f_end-f_begin)), np.zeros((f_end-f_begin),dtype=np.int64) # Rvir and Nvir of all the haloes found by the FOF
name_save, prop_full_save = [], [] # name and prop of haloes I am interested in
prop_num_save, R_and_M_save = np.zeros((N_haloes,11)), np.zeros((N_haloes,13))  # prop of haloes I am interested in
N_ok = 0 # It is incremented if I am interested by a halo
t_in = time.time() # total time of computation
Rvir = prop[:,2] # Rvir FOF
N_FOF_all = prop[:,0]
cdm = np.array((prop[:,6], prop[:,7], prop[:,8])) # halo center FOF (gravitational potential minimal point), so correct (Mpc/h)
indices_f = range(f_begin,f_end)
indices_f = sample(indices_f,f_end-f_begin)
for f in indices_f: # loop on haloes N_haloes
    print('f =',f)
    print('name f =',names[f])
    ###########################################################################
    # main computation
    halo_final, R_delta, N_delta, mass_ok, R_and_M = beyond_FOF_func(cdm, Rvir, path_cube, name_file, f, path_out, 
                                                                                                   Mass_min, Mass_max,save_data=save_data) 
    #halo_final, R_delta_all[ind], N_delta_all[ind], mass_ok = beyond_FOF_func(cdm, Rvir, path_cube, name_file, f, path_out, 
    #                                                                      Mass_min, Mass_max) 
    if mass_ok == True : # halo I am interested in
        print('ok')        
        #R_delta, N_delta = R_delta_all[f-f_begin], N_delta_all[f-f_begin] # Rvir and Nvir I am interested in
        cdm_pot_x, cdm_pot_y, cdm_pot_z = prop[f,6], prop[f,7], prop[f,8] # halo center (Mpc/h)
        cdm_fof_x, cdm_fof_y, cdm_fof_z = prop[f,3], prop[f,4], prop[f,5] # halo center (Mpc/h)
        cdm_dens_x, cdm_dens_y, cdm_dens_z = prop[f,9], prop[f,10], prop[f,11] # halo center (Mpc/h)
        name = names[f]
        prop_num_save[N_ok] = np.array((R_delta, N_delta, 
                                        cdm_fof_x, cdm_fof_y, cdm_fof_z,
                                        cdm_pot_x, cdm_pot_y, cdm_pot_z,
                                        cdm_dens_x, cdm_dens_y, cdm_dens_z))
        prop_full_save = prop_full_save + [[name,R_delta, N_delta, 
                                           cdm_fof_x, cdm_fof_y, cdm_fof_z,
                                           cdm_pot_x, cdm_pot_y, cdm_pot_z,
                                           cdm_dens_x, cdm_dens_y, cdm_dens_z]]
        name_save = name_save + [name]
        R_and_M = np.append(R_and_M,N_FOF_all[f])
        print(R_and_M)
        R_and_M_save[N_ok] = R_and_M
        N_ok += 1 # incrementation
        if N_ok == N_ok_threshold :
            break

print(R_and_M_save)
print('N_ok =',N_ok)

#R_delta_all = pd.DataFrame(R_delta_all,columns=['Rvir (Mpc/h)'])
#R_delta_all.to_csv(path_out+'halo_properties/Rvir_all_'+str(n_parallel)+'.dat',index=False)
#N_delta_all = pd.DataFrame(N_delta_all,columns=['Nvir'])
#N_delta_all.to_csv(path_out+'halo_properties/Nvir_all_'+str(n_parallel)+'.dat',index=False)
if N_ok > 0 :
    prop_num_save = prop_num_save[0:N_ok]
    prop_num_save = pd.DataFrame(prop_num_save,columns=['Rvir (Mpc/h)','Nvir',
                                                        'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                        'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                        'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z'])
    R_and_M_save = R_and_M_save[0:N_ok]
    R_and_M_save = pd.DataFrame(R_and_M_save,columns=['Rvir (Mpc/h)','Nvir','R200m','N200m','R200c','N200c',
                                                      'R500c','N500c','R1000c','N1000c','R2000c','N2000c','NFOF'])
    name_save = pd.DataFrame(name_save,columns=['halo name'])
    prop_full_save = pd.DataFrame(prop_full_save,columns=['halo name','Rvir (Mpc/h)','Nvir',
                                                          'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                          'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                          'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z'])
    if save_prop == True :
        prop_num_save.to_csv(path_out+'halo_properties/prop_saved_'+str(n_parallel)+'.dat',index=False)
        R_and_M_save.to_csv(path_out+'halo_properties/R_and_M_saved_'+str(n_parallel)+'.dat',index=False)
        name_save.to_csv(path_out+'halo_properties/names_saved_'+str(n_parallel)+'.dat',index=False)
        prop_full_save.to_csv(path_out+'halo_properties/prop_names_saved_'+str(n_parallel)+'.dat',index=False)
t_fin = time.time()
print('total time =',t_fin - t_in)




    

