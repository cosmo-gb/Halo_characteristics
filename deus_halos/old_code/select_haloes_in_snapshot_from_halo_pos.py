#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:20:02 2021

@author: guillaume
"""


"""
This code looks for paticles of a halo in the full simulation box. The goal is to go beyond the Friend of Friend (FOF) algorithm
and select ourself the particles in the full box of simulation for each halo. 
It takes the center of mass (CDM), the Rvir (computed with the data from the FOF file) and the name of each halo.
It takes as input the output of the halo_position files from DEUS, containing the mass and the position of the halos of DEUS (masst files)
It computes the boundaries of each of the 512 cubes of the simulation. Each particle belongs to one of this cube.
For each halo, with its CDM and Rvir, it assigns between 1 and 8 (in 3D) cubes, depending on if the halo is at the border of a cube.
It also checks if the cubes assigned are at the opposite side of the simulation (periodic boundary condition).
This information are computed by the two functions: 
check_halo_in_which_cube(data_properties,boundaries,factor,L_box) and find_cubes(halo_cube)
It is then given to the function: read_binary_file(name,N_particles_in_FOF,Rvir,cdm,halo_cube,factor,marge,N_cube,L_box)
"""

from os.path import isfile
from random import sample
import time
import sys
import struct
import numpy as np
import pandas as pd
import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()
# import matplotlib.pyplot as plt


save_data = False
save_prop = False
cluster = False
if cluster == True:  # case where I apply this code on the cluster
    # I import a number from a terminal command line
    n_parallel = np.int(sys.argv[1])
    redshift_str = sys.argv[2]
    cosmo = sys.argv[3]
    '''if np.int(sys.argv[3]) == 0 :
        print(np.int(sys.argv[3]))
        cosmo = 'lcdm'
    if np.int(sys.argv[3]) == 1 :
        print(np.int(sys.argv[3]))
        cosmo = 'rpcdm'
    elif np.int(sys.argv[3]) == 2 :
        print(np.int(sys.argv[3]))
        cosmo = 'wcdm'
    '''
    # mass_log = np.int(sys.argv[2]) + 0.5 # mass I look for halo in log (Msun/h)
else:  # case where I apply this code on my laptop
    cosmo = 'lcdm'
    redshift_str = '1'
    # mass_log = 13.5 # mass I look for halo in log (Msun/h)
    n_parallel = 1


file_param_name = 'param_DEUS_box_648_'+cosmo+'w7_z_'+redshift_str
file_param = __import__(file_param_name)

L_box = file_param.L_box
rho_vir_Brian_Norman = file_param.rho_vir_Brian_Norman
mass_one_particle = file_param.mass_one_particle
rho_crit_0 = file_param.rho_crit_0
Omega_matter_0 = file_param.Omega_matter_0
redshift = file_param.redshift
v_ren = file_param.v_ren
hsml = file_param.hsml


def compute_cdm_with_phi(x, y, z, ind_phi, N_mean):
    ''' 
    Compute the cdm from a list of phi function indices (potential gravitational or local density)
    The center is the center of mass around the particle with its N_mean neighbours
    '''
    cdm_x_0 = x[ind_phi[0]]  # potential minimum point
    cdm_y_0 = y[ind_phi[0]]
    cdm_z_0 = z[ind_phi[0]]
    radius = np.sqrt((x-cdm_x_0)**2 + (y-cdm_y_0)**2 + (z-cdm_z_0)**2)
    ind_radius_sorted = np.argsort(radius)
    cdm_x = np.mean(x[ind_radius_sorted[0:N_mean]])
    cdm_y = np.mean(y[ind_radius_sorted[0:N_mean]])
    cdm_z = np.mean(z[ind_radius_sorted[0:N_mean]])
    cdm_phi = np.array((cdm_x, cdm_y, cdm_z))  # halo center in Mpc/h
    return (cdm_phi)


def cdm_shrinked_sphere(x, y, z, fac_N=0.01, fac_rad=0.025, r_max=-1, N_min=100):
    ''' It computes the shrinking sphere center.
    It returns it in the frame of x, y and z and in the same unit
    '''
    # Initialisation
    N_part_initial = len(x)
    cdm = np.array((np.mean(x), np.mean(y), np.mean(z)))
    x_i = x - cdm[0]
    y_i = y - cdm[1]
    z_i = z - cdm[2]
    radius_i = np.sqrt(x_i**2 + y_i**2 + z_i**2)
    if r_max == -1:
        r_max = np.max(radius_i)
    N_part_i = len(x)
    fac_N = 0.01
    i = 0
    while (N_part_i > fac_N * N_part_initial and N_part_i > N_min):  # loop
        r_threshold_i = r_max * (1 - fac_rad)**i
        ind_i = np.where(radius_i < r_threshold_i)[0]
        x_i, y_i, z_i = x_i[ind_i], y_i[ind_i], z_i[ind_i]
        cdm_i = np.array((np.mean(x_i), np.mean(y_i), np.mean(z_i)))
        x_i_use = x_i - cdm_i[0]
        y_i_use = y_i - cdm_i[1]
        z_i_use = z_i - cdm_i[2]
        radius_i = np.sqrt(x_i_use**2 + y_i_use**2 + z_i_use**2)
        N_part_i = len(ind_i)
        i += 1
    cdm_shrink = cdm_i + cdm
    return (cdm_shrink)


def compute_centers(x, y, z, N_mean=32):
    ''' This function computes the halo center by computing the center of mass inside
    the area of the halo centered on the minimum gravitational potential, containg N_mean particles.
    It is the potential center of Neto_07: https://ui.adsabs.harvard.edu/abs/2007MNRAS.381.1450N/abstract
    It also computes the highest density point cdm_dens in a similar way.
    It also computes the basic center of mass in the FOF data cdm_fof
    And finally it also computes the shrinking sphere center
    In order to compute the potential gravitational, Dehnen function getGravity is used
    Details on this function can be found here:  https://pypi.org/project/python-unsiotools/
    and by writing in a terminal: pydoc unsiotools.simulations.cfalcon
    Be carefull, Dehnen function can give different results dependending of the frame used,
    e.g. if I reset x,y,z in another frame (by adding a coonstant) it can change the results of Dehnen
    All the cdms are returned in Mpc/h, in the frame of x,y,z
    '''
    cdm_fof = np.array((np.mean(x), np.mean(y), np.mean(z))
                       )  # standard center of mass in Mpc/h
    # I reset the particle position in this approximate center frame
    x_use, y_use, z_use = x - cdm_fof[0], y - cdm_fof[1], z - cdm_fof[2]
    N_part = len(x_use)  # number of particles
    # I reset the particle positions in order to give it as input for Dehnen
    pos = np.zeros((N_part, 3), dtype=np.float32)
    pos[:, 0], pos[:, 1], pos[:, 2] = x_use, y_use, z_use
    pos = np.reshape(pos, N_part*3)
    mass_array = np.ones((N_part), dtype=np.float32)
    ok, acc, pot = cf.getGravity(pos, mass_array, hsml)
    # with the minimum of the potential gravitational, I compute the halo center: cdm_pot
    ind_pot = np.argsort(pot)
    cdm_pot_use = compute_cdm_with_phi(x_use, y_use, z_use, ind_pot, N_mean)
    cdm_pot = cdm_pot_use + cdm_fof
    # with the local density, I compute the highest density point: cdm_dens
    ok, rho_one_particle, h_sml = cf.getDensity(pos, mass_array)
    ind_rho = np.argsort(rho_one_particle)
    ind_rho = ind_rho[::-1]  # reverse the order: put the densest first
    cdm_dens_use = compute_cdm_with_phi(x_use, y_use, z_use, ind_rho, N_mean)
    cdm_dens = cdm_dens_use + cdm_fof
    # shrinking sphere
    cdm_shrink_use = cdm_shrinked_sphere(x_use, y_use, z_use)
    cdm_shrink = cdm_shrink_use + cdm_fof
    # distances between pot and dens in Mpc/h
    dis_pot_dens = np.sqrt(
        (cdm_pot[0]-cdm_dens[0])**2 + (cdm_pot[1]-cdm_dens[1])**2 + (cdm_pot[2]-cdm_dens[2])**2)
    dis_pot_shrink = np.sqrt((cdm_pot[0]-cdm_shrink[0])**2 + (
        cdm_pot[1]-cdm_shrink[1])**2 + (cdm_pot[2]-cdm_shrink[2])**2)
    return (cdm_fof, cdm_pot, cdm_dens, cdm_shrink, dis_pot_dens, dis_pot_shrink)


def spherical_selection(x, y, z, R, cdm):
    # set in the center of mass frame
    radius = np.sqrt((x - cdm[0])**2 + (y - cdm[1])**2 + (z - cdm[2])**2)
    ind_sel = np.where(radius < R)[0]
    return (ind_sel)


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
    N_part_array = np.linspace(1, N_part_tot, N_part_tot)
    rho = N_part_array * 3/(4*np.pi*(r_sorted)**3)
    # I take the last indice rho > rho_lim
    ind_rho = np.where(rho > rho_delta)[0]
    error_0 = 0
    error_fof = 0
    N_delta = len(ind_rho)
    if N_delta > 0 and N_delta < N_part_tot:
        R_delta = (r_sorted[N_delta] + r_sorted[N_delta-1])/2
    elif N_delta == 0:
        # the threshold is too large and there is no particles
        # with higher density than that threshold
        R_delta = 0
        error_0 += 1
        print('''the density threshold is too large, \n 
              you have no particles inside this threshold''')
    elif N_delta == N_part_tot:
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
        R_delta = -1  # Big problem !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(
            'N_delta should be in the interval [0,N_part_tot] and it is not the case')
    # compute also M_200_m, M_200_c, M_500_c, M_1000_c, M_2000_c
    # https://iopscience.iop.org/article/10.1088/0004-637X/792/1/25/pdf
    rho_200_m = 200 * rho_crit_0 * Omega_matter_0 * (1 + redshift)**3
    N_200_m = len(np.where(rho > rho_200_m)[0])
    if N_200_m > 0 and N_200_m < N_part_tot:
        R_200_m = (r_sorted[N_200_m] + r_sorted[N_200_m-1])/2
    elif N_200_m == 0:
        R_200_m = 0
    elif N_200_m == N_part_tot:
        R_200_m = (3 * N_part_tot/(4 * np.pi * rho_200_m))**(1/3)
    else:
        R_200_m = -1
    # 200_c
    Omega_c = 1-Omega_matter_0 + Omega_matter_0 * (1 + redshift)**3
    rho_200_c = 200 * rho_crit_0 * Omega_c
    N_200_c = len(np.where(rho > rho_200_c)[0])
    if N_200_c > 0 and N_200_c < N_part_tot:
        R_200_c = (r_sorted[N_200_c] + r_sorted[N_200_c-1])/2
    elif N_200_c == 0:
        R_200_c = 0
    elif N_200_c == N_part_tot:
        R_200_c = (3 * N_part_tot/(4 * np.pi * rho_200_c))**(1/3)
    else:
        R_200_c = -1
    # 500_c
    rho_500_c = 500 * rho_crit_0 * Omega_c
    N_500_c = len(np.where(rho > rho_500_c)[0])
    if N_500_c > 0 and N_500_c < N_part_tot:
        R_500_c = (r_sorted[N_500_c] + r_sorted[N_500_c-1])/2
    elif N_500_c == 0:
        R_500_c = 0
    elif N_500_c == N_part_tot:
        R_500_c = (3 * N_part_tot/(4 * np.pi * rho_500_c))**(1/3)
    else:
        R_500_c = -1
    # 1000_c
    rho_1000_c = 1000 * rho_crit_0 * Omega_c
    N_1000_c = len(np.where(rho > rho_1000_c)[0])
    if N_1000_c > 0 and N_1000_c < N_part_tot:
        R_1000_c = (r_sorted[N_1000_c] + r_sorted[N_1000_c-1])/2
    elif N_1000_c == 0:
        R_1000_c = 0
    elif N_1000_c == N_part_tot:
        R_1000_c = (3 * N_part_tot/(4 * np.pi * rho_1000_c))**(1/3)
    else:
        R_1000_c = -1
    # 2000_c
    rho_2000_c = 2000 * rho_crit_0 * Omega_c
    N_2000_c = len(np.where(rho > rho_2000_c)[0])
    if N_2000_c > 0 and N_2000_c < N_part_tot:
        R_2000_c = (r_sorted[N_2000_c] + r_sorted[N_2000_c-1])/2
    elif N_2000_c == 0:
        R_2000_c = 0
    elif N_2000_c == N_part_tot:
        R_2000_c = (3 * N_part_tot/(4 * np.pi * rho_2000_c))**(1/3)
    else:
        R_2000_c = -1
    ###########################################################################
    output_R_and_M = np.array((R_delta, N_delta, R_200_m, N_200_m,
                               R_200_c, N_200_c, R_500_c, N_500_c,
                               R_1000_c, N_1000_c, R_2000_c, N_2000_c))
    output_number = np.array((R_delta, N_delta, error_0, error_fof))
    output_array = np.array((r_sorted, ind, radius, rho))
    return (output_number, output_array, output_R_and_M)


def check_halo_in_which_cube(boundaries, cdm_i, Rlim, halo_cube, i, L_box):
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
    for c_i in range(N_cubes_1D):  # loop on the cubes inside the full box in the i direction
        # boundaries of the cube c_i in the i direction
        bound_min = boundaries[c_i]
        bound_max = boundaries[c_i+1]
        ind_cdm_i_sup = np.where(cdm_i > bound_min)[0]
        ind_cdm_i_inf = np.where(cdm_i[ind_cdm_i_sup] < bound_max)[0]
        # contains only haloes inside the cube c_i
        ind_i = ind_cdm_i_sup[ind_cdm_i_inf]
        cdm_i_used = cdm_i[ind_i]
        Rlim_used = Rlim[ind_i]
        N_c_i = len(ind_i)
        i_n_used = np.zeros((N_c_i), dtype=int)
        ind_plus = np.where(cdm_i_used - Rlim_used <= bound_min)[0]
        i_n_used[ind_plus] = -1  # lower boundary effect possible
        ind_minus = np.where(cdm_i_used + Rlim_used >= bound_max)[0]
        i_n_used[ind_minus] = 1  # upper boundary effect possible
        c_i_array = np.ones((N_c_i), dtype=int) * c_i
        halo_cube[ind_i, 0], halo_cube[ind_i,
                                       i], halo_cube[ind_i, i+3] = ind_i, c_i_array, i_n_used
    return (halo_cube)


def find_cubes(halo_cube, N_cubes_1D=8):
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
    n = 0  # number of direction I need to shift, to look for in another cube
    # the number 137 means that nothing needs to be done ine the i direction,
    N_fun = 137  # N_fun could be anything not between [-1,1]
    h = halo_cube[0]  # halo numero
    # cube numero where the halo center cdm_i is in the i direction
    c_x, c_y, c_z = halo_cube[1], halo_cube[2], halo_cube[3]
    # need to look for another cube in the i direction (i_n=-1 or 1) or not (i_n=0)
    x_n, y_n, z_n = halo_cube[4], halo_cube[5], halo_cube[6]
    # I begin by considering the cube of the halo center, initialization
    # if all the i_n=0, I just add [n] to that list and return it
    halo = [[h], [c_x, c_y, c_z, N_fun, N_fun, N_fun]]  # no shift case, 1 cube
    # cube numero of the cube neighbour in the i direction
    c_x_new, c_y_new, c_z_new = (
        c_x + x_n) % N_cubes_1D, (c_y + y_n) % N_cubes_1D, (c_z + z_n) % N_cubes_1D
    # The i_n_new's won't be used in the direction where i_n=0, and I will use N_fun in that direction
    # if used, it is =0 if the cube containing the halo center is not close to the boundary
    # of the full simulation box. Otherwise, it is =-1 or =1.
    x_n_new, y_n_new, z_n_new = (
        c_x + x_n)//N_cubes_1D, (c_y + y_n)//N_cubes_1D, (c_z + z_n)//N_cubes_1D
    # potentially 7 more cubes than the one of the halo center, 1 cube by if
    if x_n != 0:  # at least 2 cubes (shift on x), but maybe more
        halo = halo + [[c_x_new, c_y, c_z, x_n_new, N_fun, N_fun]]
        n += 1  # I need to shift in the x direction
    if y_n != 0:  # at least 2 cubes (shift on y), but maybe more
        halo = halo + [[c_x, c_y_new, c_z, N_fun, y_n_new, N_fun]]
        n += 1  # I need to shift in the y direction
    if z_n != 0:  # at least 2 cubes (shift on z), but maybe more
        halo = halo + [[c_x, c_y, c_z_new, N_fun, N_fun, z_n_new]]
        n += 1  # I need to shift in the z direction
    # if I need more than 1 cube, I also need to consider the shift in both directions
    # at least 4 cubes (shift on x and y), but maybe more
    if x_n != 0 and y_n != 0:
        halo = halo + [[c_x_new, c_y_new, c_z, x_n_new, y_n_new, N_fun]]
    # at least 4 cubes (shift on x and z), but maybe more
    if x_n != 0 and z_n != 0:
        halo = halo + [[c_x_new, c_y, c_z_new, x_n_new, N_fun, z_n_new]]
    # at least 4 cubes (shift on y and z), but maybe more
    if y_n != 0 and z_n != 0:
        halo = halo + [[c_x, c_y_new, c_z_new, N_fun, y_n_new, z_n_new]]
    # 8 cubes (maximum) (shift on x, y and z)
    if x_n != 0 and y_n != 0 and z_n != 0:
        halo = halo + [[c_x_new, c_y_new, c_z_new, x_n_new, y_n_new, z_n_new]]

    halo = halo + [[n]]
    return (halo)


def select_particles_in_cube(x, y, z, Rlim, cdm):
    '''This function selects particles in a cube centered on cdm and of half size Rlim.
    It takes the particle positions x,y,z (in Mpc/h), velocities vx,vy,vz, and identities ide.
    It selects those which are in the cube I want (defined with cdm and Rlim).
    It returns the positions, velocities and identities of those particles, with their number N_part.
    Be carefull, it does not check the periodicity !!!
    '''
    ind_x = np.where(np.abs(x - cdm[0]) < Rlim)[0]  # particles with good x's
    y_new_x, z_new_x = y[ind_x], z[ind_x]
    # particles with good x's and y's
    ind_x_y = np.where(np.abs(y_new_x - cdm[1]) < Rlim)[0]
    z_new_x_y = z_new_x[ind_x_y]
    # particles with good x's, y's and z's
    ind_x_y_z = np.where(np.abs(z_new_x_y - cdm[2]) < Rlim)[0]
    ind = ind_x[ind_x_y][ind_x_y_z]  # particles with good x's, y's and z's
    # particles number which are in the cube that I want to select
    N_part = len(ind)
    x, y, z = x[ind], y[ind], z[ind]
    return (x, y, z, ind, N_part)


def periodicity(x, y, z, cdm, dis_1=0.5, dis_2=0.9, box=1):
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
    distance = np.sqrt((x-cdm[0])**2 + (y-cdm[1])**2 +
                       (z-cdm[2])**2)  # distance from the center in box unit
    # where the distance is VERY large
    indice_problem = np.where(distance > dis_1)[0]
    if len(indice_problem) > 0:
        bad_periodicity = 1  # there is a periodicity problem
    else:
        bad_periodicity = 0  # there is no periodicity problem
    if bad_periodicity == 1:  # then I correct the problem
        # I deal with the problem in the 3 direction of space separetely
        distance_x, distance_y, distance_z = np.abs(
            x - cdm[0]), np.abs(y - cdm[1]), np.abs(z - cdm[2])
        ind_x, ind_y, ind_z = np.where(distance_x > dis_1)[0], np.where(
            distance_y > dis_1)[0], np.where(distance_z > dis_1)[0]
        # contains the problematic positions
        my_x, my_y, my_z = x[ind_x], y[ind_y], z[ind_z]
        N_x, N_y, N_z = len(my_x), len(my_y), len(
            my_z)  # number of problematic positions
        # I first check if the problematic positions are > dis_2 (thought as close to 1),
        # and I remove box for those where it is the case.
        # Then I just add 1 for the problematic case which are not > dis_2
        ind_x_minus, ind_y_minus, ind_z_minus = np.where(
            my_x > dis_2)[0], np.where(my_y > dis_2)[0], np.where(my_z > dis_2)[0]
        my_x[ind_x_minus], my_y[ind_y_minus], my_z[ind_z_minus] = my_x[ind_x_minus] - \
            box, my_y[ind_y_minus] - box, my_z[ind_z_minus] - box
        ind_x_plus, ind_y_plus, ind_z_plus = np.ones(N_x, dtype=bool), np.ones(
            N_y, dtype=bool), np.ones(N_z, dtype=bool)
        ind_x_plus[ind_x_minus], ind_y_plus[ind_y_minus], ind_z_plus[ind_z_minus] = False, False, False
        my_x[ind_x_plus], my_y[ind_y_plus], my_z[ind_z_plus] = my_x[ind_x_plus] + \
            box, my_y[ind_y_plus] + box, my_z[ind_z_plus] + box
        # I replace the corrected particles
        x[ind_x], y[ind_y], z[ind_z] = my_x, my_y, my_z

    return (x, y, z, bad_periodicity)


def read_binary_file(path_cube, name_file, halo_cube, cdm, Rvir, Rlim, N_marge, N_cubes, L_box):
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
    # the 0 will be replaced by 1 if there are indeed particles in the cube c
    found_part = np.zeros((8), dtype=int)
    # in order to check if I do not add zeroes in my output
    N_zeros_x, N_zeros_y, N_zeros_z = 0, 0, 0
    # I choose positions and velocities to be np.float64 in order to avoid duplicates on positions
    # But in http://www.deus-consortium.org/a-propos/cosmological-models/post-processing/,
    # they say that it is simple precision, meaning np.float32
    # I choose identities to be np.int64, according to deus-consortium.
    my_type_pos, my_type_vel, my_type_id = np.float64, np.float64, np.int64  # data type
    # I will set the positions, velocities and identities of the particles
    # I am interested in, in the following 3 arrays called pos, vel and ide.
    # N_marge must be larger than the number of particles I want to select
    # (I do not know this number à priori, so I take a marge)
    pos = np.zeros((N_marge, 3), dtype=my_type_pos)
    vel = np.zeros((N_marge, 3), dtype=my_type_vel)
    ide = np.zeros((N_marge), dtype=my_type_id)
    N_tot = 0  # total particle number that I want to select, initialized at 0
    # number of different cubes I need to explore, in order to look for particles
    # N_cube_loop is an integer between [1,8]
    N_cube_loop = np.int(2**halo_cube[-1][0])
    # First I set the cdm in the 3 directions between [0,L_box]
    # because the output of DEUS are also between [0,L_box]
    # ind_cdm_plus contains the information of the upper boundary problem
    ind_cdm_plus = np.where(cdm > L_box)[0]
    if len(ind_cdm_plus) != 0:  # at least 1 upper boundary problem
        cdm[ind_cdm_plus] = cdm[ind_cdm_plus] - L_box
    # ind_cdm_moins contains the information of the lower boundary problem
    ind_cdm_moins = np.where(cdm < 0)[0]
    if len(ind_cdm_moins) != 0:  # at least 1 lower boundary problem
        cdm[ind_cdm_moins] = cdm[ind_cdm_moins] + L_box
    # loop on the different cubes I want to explore, in order to look for particles
    for c in range(N_cube_loop):
        # here we study the case of the cube c, characterized by the c_i's
        # I select which c_i's I will use and the corresponding i_n's, in my_cube
        # which then contains 6 elements: 3 c_i's and 3 i_n's
        my_cube = halo_cube[c+1]
        # c_x, c_y and c_z integers between [0,N_cube_1D-1]
        c_i = np.array((my_cube[0], my_cube[1], my_cube[2]))
        # x_n, y_n and z_n integers between [-1,1] or 137
        i_n = np.array((my_cube[3], my_cube[4], my_cube[5]))
        # cube numbers in 1 direction in the simulation box
        N_cube_1D = np.int(np.cbrt(N_cubes))
        # numero of the file corresponding to 1 cube c_x, c_y, c_z, integer between [0,N_cubes-1]
        num = np.int(c_i[0]*N_cube_1D**2 + c_i[1]*N_cube_1D + c_i[2])
        # I open the binary file numero num, it corresponds to 1 of the N_cubes of the simulation
        file = open(path_cube+name_file+'_' +
                    str('{:d}'.format(num).zfill(5))+"", "rb")
        # number of bytes for the total number of particles
        struct.unpack('<i', file.read(4))
        # total number of particles in the cube c
        N_part_tot = struct.unpack('<i', file.read(4))[0]
        # number of bytes of procid, the identity of the binary file
        struct.unpack('<i', file.read(4))
        # procid, the identity of the binary file
        struct.unpack('<i', file.read(4))[0]
        # number of bytes of the boundaries for the position
        struct.unpack('<iii', file.read(4*3))
        # boundaries for the position in box=1 unit
        struct.unpack('<'+str(6)+'f', file.read(4*6))
        struct.unpack('<ii', file.read(8))  # number of bytes for the position
        # positions of all particles in the file, it lies between [0,1] (box unit)
        pos_cube = struct.unpack(
            '<'+str(N_part_tot*3)+'f', file.read(4*3*N_part_tot))
        # number of bytes for the velocities
        struct.unpack('<ii', file.read(8))
        # velocities of all particles in the file, in simu unit
        vel_cube = struct.unpack(
            '<'+str(N_part_tot*3)+'f', file.read(4*3*N_part_tot))
        # number of bytes for the identities
        struct.unpack('<ii', file.read(8))
        # identities of all particles in the file, integers between[1,N_tot_simu]
        ide_cube = struct.unpack(
            '<'+str(N_part_tot)+'q', file.read(8*N_part_tot))
        pos_cube = np.array(pos_cube) * L_box  # I set the positions in Mpc/h
        # I reshape (N_part_tot lines and 3 columns)
        pos_cube = np.reshape(pos_cube, (N_part_tot, 3))
        x, y, z = pos_cube[:, 0], pos_cube[:, 1], pos_cube[:, 2]
        vel_cube = np.array(vel_cube)
        vel_cube = np.reshape(vel_cube, (N_part_tot, 3))
        ide_cube = np.array(ide_cube, dtype=my_type_id)
        # I look for zeroes which are really in the simulation (it is a possible position)
        ind_x_zeros, ind_y_zeros, ind_z_zeros = np.where(x == 0)[0], np.where(y == 0)[
            0], np.where(z == 0)[0]
        N_zeros_x, N_zeros_y, N_zeros_z = N_zeros_x + \
            len(ind_x_zeros), N_zeros_y + \
            len(ind_y_zeros), N_zeros_z + len(ind_z_zeros)
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
        x_new, y_new, z_new, ind_new, N_part_new = select_particles_in_cube(
            x, y, z, Rlim, cdm)
        pos_new = np.array((x_new, y_new, z_new)).transpose()
        vel_new, ide_new = vel_cube[ind_new, :], ide_cube[ind_new]
        if N_part_new > 0:
            # If N_part_new > 0, it means that I indeed found particles inside the cube c
            # which is expected, otherwise, it is very likely that there is a problem.
            # either the cube is not the right where I need to look for.
            # But it is also possible that it is a limit issue, meaning that
            # the volume where I need to look for inside the cube c is so small
            # that there are no particles in it. Again, it is very unlikely.
            found_part[c] = 1
        N_beg, N_end = N_tot, N_tot + N_part_new
        if N_end > N_marge:
            print('The number of particles found N_tot is greater than N_tot > ', N_end - 1, '. \n '
                  'And so it is superior to the number of particles allowed N_marge = ', N_marge, '. \n',
                  'You should increase N_marge.')
            break
        pos[N_beg:N_end, :], vel[N_beg:N_end,
                                 :], ide[N_beg:N_end] = pos_new, vel_new, ide_new
        cdm[ind_new_cdm_moins] = cdm[ind_new_cdm_moins] - \
            L_box  # I replace the cdm
        cdm[ind_new_cdm_plus] = cdm[ind_new_cdm_plus] + L_box
        N_tot = N_tot + N_part_new  # new number of particles that I am looking for
    # end loop cube
    pos, vel, ide = pos[0:N_tot, :]/L_box, vel[0:N_tot, :], ide[0:N_tot]
    x_new, y_new, z_new = pos[:, 0], pos[:, 1], pos[:, 2]
    x_new, y_new, z_new, bad_periodicity = periodicity(
        x_new, y_new, z_new, cdm/L_box)
    x_new, y_new, z_new = x_new*L_box, y_new*L_box, z_new*L_box
    ind_sel = spherical_selection(x_new, y_new, z_new, Rvir, cdm)
    # I compute the centers inside the sphere centered on the cdm given by the FOF and of radius given by the FOF mass (BrianNorman)
    cdm_all = compute_centers(x_new[ind_sel], y_new[ind_sel], z_new[ind_sel])
    out_num, out_array, out_R_and_M = compute_R_delta(
        cdm_all[1], x_new, y_new, z_new)
    R_delta, N_delta = out_num[0], out_num[1]
    vx_new, vy_new, vz_new = vel[:, 0], vel[:, 1], vel[:, 2]
    halo_final = np.array(
        (x_new/L_box, y_new/L_box, z_new/L_box, vx_new, vy_new, vz_new, ide))
    ind_sphere_delta = spherical_selection(x_new, y_new, z_new, R_delta, cdm)
    halo_sphere_delta = np.array((x_new[ind_sphere_delta], y_new[ind_sphere_delta], z_new[ind_sphere_delta],
                                  vx_new[ind_sphere_delta]*v_ren, vy_new[ind_sphere_delta] *
                                  v_ren, vz_new[ind_sphere_delta]*v_ren,
                                  ide[ind_sphere_delta]))
    ind_pb_x, ind_pb_y, ind_pb_z = np.where(x_new == 0)[0], np.where(y_new == 0)[
        0], np.where(z_new == 0)[0]
    halo_final_pb = 0  # check if there are too many zeros positions in the particles
    if len(ind_pb_x) > N_zeros_x or len(ind_pb_y) > N_zeros_y or len(ind_pb_z) > N_zeros_z:
        halo_final_pb = 1  # there are too many zeros positions in the particles
    # characterized in which cube I found particles
    ind_found_part = np.where(found_part != 0)[0]

    return (halo_final, halo_sphere_delta, ind_found_part, N_cube_loop, halo_final_pb, R_delta, N_delta, out_R_and_M, cdm_all)


def check_if_file_exist(path_cube, name_file, halo_cube, N_cubes):
    # N_cube_loop is an integer between [1,8]
    N_cube_loop = np.int(2**halo_cube[-1][0])
    # 0 = the file does not exist
    file_existence = np.zeros(N_cube_loop, dtype=int)
    for c in range(N_cube_loop):
        my_cube = halo_cube[c+1]
        # c_x, c_y and c_z integers between [0,N_cube_1D-1]
        c_i = np.array((my_cube[0], my_cube[1], my_cube[2]))
        # cube numbers in 1 direction in the simulation box
        N_cube_1D = np.int(np.cbrt(N_cubes))
        # numero of the file corresponding to 1 cube c_x, c_y, c_z, integer between [0,N_cubes-1]
        num = np.int(c_i[0]*N_cube_1D**2 + c_i[1]*N_cube_1D + c_i[2])
        if c == 0:
            num_main = num
        # print('num =',num)
        file_c_ex = isfile(path_cube+name_file+'_' +
                           str('{:d}'.format(num).zfill(5)))
        if file_c_ex == True:
            file_existence[c] = 1  # the file c exists
    return (file_existence, num_main)


def beyond_FOF_func(cdm, Rvir, path_cube, name_file, f, name_halo_f, path_out,
                    L_box=L_box, N_cubes=512, factor=5, marge=100, save_data=True):
    # Rlim used (Mpc/h) (half size of the cube where I want to take particles)
    Rlim = factor * Rvir
    N_loop = np.int(np.cbrt(N_cubes))  # number of cubes **(1/3) = N_cubes_1D
    # boundaries of the cubes in the simulation in 1D
    boundaries = np.linspace(0, L_box, N_loop+1, dtype=float)
    N_haloes = len(Rlim)  # number of haloes
    # halo_cube contains : halo numero, cubes c1,c2,c3 numeros, boundaries x_n,y_n,z_n,
    # see the function check_halo_in_which_cube for details
    halo_cube = np.zeros((N_haloes, 7), dtype=int)
    print('L_box =', L_box)
    # print('cdm =',cdm)
    halo_cube = check_halo_in_which_cube(
        boundaries, cdm[0], Rlim, halo_cube, 1, L_box)
    halo_cube = check_halo_in_which_cube(
        boundaries, cdm[1], Rlim, halo_cube, 2, L_box)
    halo_cube = check_halo_in_which_cube(
        boundaries, cdm[2], Rlim, halo_cube, 3, L_box)
    my_halo_cube = halo_cube[f]
    my_cubes = find_cubes(my_halo_cube)
    print(my_cubes)
    cdm_f = np.array((cdm[0, f], cdm[1, f], cdm[2, f]))  # halo center (Mpc/h)
    Rlim_f = Rlim[f]
    N_marge = np.int(prop[f, 1]) * marge
    file_existence, num_main = check_if_file_exist(
        path_cube, name_file, my_cubes, N_cubes)
    print('num_main =', num_main)
    if np.min(file_existence) == 1:  # it means that all the needed files exist
        print('The needed files do exist')
        halo_final_all = read_binary_file(
            path_cube, name_file, my_cubes, cdm_f, Rvir[f], Rlim_f, N_marge, N_cubes, L_box)
        file_exist_pb = False
    else:
        print('The needed files do not exist')
        save_data = False
        halo_final_all = np.zeros((9))
        halo_final_all[3] = 1
        file_exist_pb = True
    # data and problem
    halo_final, halo_sphere_delta, halo_final_pb = halo_final_all[
        0], halo_final_all[1], halo_final_all[4]
    R_delta, N_delta = halo_final_all[5], halo_final_all[6]  # Rvir and Nvir
    out_R_and_M = halo_final_all[7]
    cdm_all = halo_final_all[8]
    mass_ok = True
    if save_data == True:  # the halo has the good mass
        print(name_halo_f)
        mass_ok = True
        halo_final = halo_final.transpose()  # reshape((N_part_tot,7))
        halo_sphere_delta = halo_sphere_delta.transpose()
        # np.savetxt(path_out+'data/'+name_f+'_'+str(factor)+'_Rvir_new.dat',halo_final)
        if save_data == True:
            halo_final_csv = pd.DataFrame(halo_final, columns=['x (DEUS unit)', 'y (DEUS unit)', 'z (DEUS unit)',
                                                               'v_x (DEUS unit)', 'v_y (DEUS unit)', 'v_z (DEUS unit)', 'ide'])
            halo_final_csv.to_csv(
                path_out+'data/'+name_halo_f+'_'+str(factor)+'_Rvir.dat', index=False)
            halo_sphere_delta_csv = pd.DataFrame(halo_sphere_delta, columns=['x (Mpc/h)', 'y', 'z',
                                                                             'v_x (km/s in box frame not in halo s)', 'v_y', 'v_z', 'ide'])
            halo_sphere_delta_csv.to_csv(
                path_out+'data/'+name_halo_f+'_sphere_1_Rvir.dat', index=False)
    if halo_final_pb != 0:
        print(name_halo_f)
        print('problem: too many zeros in the particle positions')
        N_zeroes_saved = pd.DataFrame(np.array([name_halo_f, halo_final_pb]), columns=[
                                      'halo name', 'number of zeroes in the final data'])
        N_zeroes_saved.to_csv(path_out+'Problem/zeroes_' +
                              name_halo_f+'.dat', index=False)
    ind_found_part, N_cube_loop = halo_final_all[2], halo_final_all[3]
    if N_cube_loop != ind_found_part.size:
        print(name_halo_f)
        print('problem: the number of cubes where I effectively found particles is not equal \n'
              'to the number of cubes I looked for particles. It is not necessarly a mistake, it is just unlikely.')
        print(N_cube_loop)
        print(ind_found_part)
        print('test')
        cube_pb = np.array(
            (f, N_cube_loop, ind_found_part.size)).reshape((1, 3))
        print(cube_pb)
        cubes_pb_saved = pd.DataFrame(cube_pb, columns=[
                                      'f (halo candidate from FOF numero)', 'Number of cubes I need to look for particles', 'Number of cubes where I found particles'])
        cubes_pb_saved.to_csv(path_out+'Problem/cubes_' +
                              name_halo_f+'.dat', index=False)
    if file_exist_pb == True:
        print(list([name_halo_f, num_main]))
        print('There is a file existence issue !')
        file_issue_saved = pd.DataFrame(
            list([num_main]), columns=['num main cube'])
        file_issue_saved.to_csv(
            path_out+'Problem/file_issue_'+name_halo_f+'.dat', index=False)
    return (halo_final, R_delta, N_delta, mass_ok, out_R_and_M, cdm_all, file_exist_pb)


def indice_cluster(n_parallel, N_haloes, N_cores):
    # if I want to apply to the cluster
    # N_haloes = 2876 # Number of haloes
    # n_use = N_haloes // N_cut
    multiple = np.int(N_haloes//(N_cores-1))
    print('N_cores =', N_cores)
    print('N_haloes =', N_haloes)
    print('multiple =', multiple)
    # multiple = 10
    if n_parallel < N_cores:
        # n_use #np.int(multiple*(n-1)) # begining and end of my loop
        f_begin = (n_parallel - 1) * multiple
        f_end = n_parallel * multiple  # n_use #np.int(multiple*n)
    else:
        f_begin = (n_parallel - 1) * multiple  # n_use #multiple*N_cores
        f_end = N_haloes
    print('f_begin,f_end =', f_begin, f_end)
    return (f_begin, f_end)


##################################################################################
# path for load properties that I get from FOF
path_prop_halo_pos = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_' + \
    redshift_str+'/'+cosmo+'/halos_position/halo_properties/'
# path for load binary data
if cluster == True:
    path_cube = '/net/nas10c/data/gbonnet/DEUS_snapshot/z_'+redshift_str+'/'+cosmo+'/'
else:
    path_cube = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_' + \
        redshift_str+'/'+cosmo+'/snapshot/'  # path where I look for the binary data
# path for save the data
path_out = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_' + \
    redshift_str+'/'+cosmo+'/most_massive/'
##################################################################################
prop_name = 'halo_pos_pre_sel.dat'
prop = np.loadtxt(path_prop_halo_pos+prop_name)
N_haloes = len(prop)
print('N_haloes =', N_haloes)
N_cores = 128
f_begin, f_end = indice_cluster(n_parallel, N_haloes, N_cores)
# f_begin, f_end = 0, 1 #64, 65
name_file = 'fof_boxlen648_n2048_'+cosmo+'w7_cube'  # name of file
name_save, prop_full_save = [], []  # name and prop of haloes I am interested in
prop_num_save, R_and_M_save = np.zeros((N_haloes, 17)), np.zeros(
    (N_haloes, 13))  # prop of haloes I am interested in
N_ok = 0  # It is incremented if I am interested by a halo
t_in = time.time()  # total time of computation
print(prop)
N_FOF_all = prop[:, 1]
Rvir = np.power(N_FOF_all * 3/(4 * np.pi * rho_vir_Brian_Norman), 1/3)
# halo center FOF (gravitational potential minimal point), so correct (Mpc/h)
cdm = np.array((prop[:, 2], prop[:, 3], prop[:, 4]))
cube_num, halo_num = np.array(
    prop[:, 5], dtype=int), np.array(prop[:, 6], dtype=int)
fac_acc = np.int(512/N_cores)
print('fac_acc =', fac_acc)
cube_min = (n_parallel-1)*fac_acc
cube_max = (n_parallel)*fac_acc
print('cube_min =', cube_min, 'cube_max =', cube_max)
# indices_f = np.where( (cube_num >= cube_min) & (cube_num < cube_max) )[0]
'''#########################################################################################
factor = 5
N_cubes = 512
#num_main = np.zeros(len(indices_f),dtype=int)
turn = 0
for f in indices_f :
    Rlim = factor * Rvir # Rlim used (Mpc/h) (half size of the cube where I want to take particles)
    N_loop = np.int(np.cbrt(N_cubes)) # number of cubes **(1/3) = N_cubes_1D
    boundaries = np.linspace(0,L_box,N_loop+1,dtype=float) # boundaries of the cubes in the simulation in 1D
    N_haloes = len(Rlim) # number of haloes
    # halo_cube contains : halo numero, cubes c1,c2,c3 numeros, boundaries x_n,y_n,z_n, 
    # see the function check_halo_in_which_cube for details
    halo_cube = np.zeros((N_haloes,7),dtype=int) 
    #print('L_box =',L_box)
    #print('cdm =',cdm)
    halo_cube = check_halo_in_which_cube(boundaries,cdm[0],Rlim,halo_cube,1,L_box)
    halo_cube = check_halo_in_which_cube(boundaries,cdm[1],Rlim,halo_cube,2,L_box)
    halo_cube = check_halo_in_which_cube(boundaries,cdm[2],Rlim,halo_cube,3,L_box)
    my_halo_cube = halo_cube[f]
    my_cubes = find_cubes(my_halo_cube) 
    #print(my_cubes)
    cdm_f = np.array((cdm[0,f],cdm[1,f],cdm[2,f])) # halo center (Mpc/h)
    Rlim_f = Rlim[f]
    #N_marge = np.int(prop[f,1]) * marge
    file_existence, num_main = check_if_file_exist(path_cube, name_file, my_cubes,N_cubes)
    turn += 1
    if num_main == 90 :
        break
print(turn)
print(f)
print(num_main)
#turn = 561
#print(np.where(num_main == 52))'''
###################################################################################
indices_f = range(f_begin, f_end)
print(indices_f)
for f in indices_f[0:4]:  # loop on haloes N_haloes
    # break
    print('f =', f)
    # print('name f =',names[f])
    halo_name_f = name_file+'_' + \
        str('{:d}'.format(cube_num[f]).zfill(5)) + \
        '_'+str('{:d}'.format(halo_num[f]).zfill(5))
    print('name f =', halo_name_f)
    ###########################################################################
    # main computation
    print('cdm_f =', cdm[:, f])
    halo_final, R_delta, N_delta, mass_ok, R_and_M, cdm_all, file_exist_pb = beyond_FOF_func(cdm, Rvir, path_cube, name_file, f, halo_name_f, path_out,
                                                                                             save_data=save_data)
    if file_exist_pb == False:  # halo I am interested in
        print('ok')
        print(cdm_all)
        # R_delta, N_delta = R_delta_all[f-f_begin], N_delta_all[f-f_begin] # Rvir and Nvir I am interested in
        cdm_cdm_x, cdm_cdm_y, cdm_cdm_z = cdm_all[0]
        # prop[f,6], prop[f,7], prop[f,8] # halo center (Mpc/h)
        cdm_pot_x, cdm_pot_y, cdm_pot_z = cdm_all[1]
        # halo center (Mpc/h)
        cdm_fof_x, cdm_fof_y, cdm_fof_z = prop[f, 2], prop[f, 3], prop[f, 4]
        # prop[f,9], prop[f,10], prop[f,11] # halo center (Mpc/h)
        cdm_dens_x, cdm_dens_y, cdm_dens_z = cdm_all[2]
        cdm_shrink_x, cdm_shrink_y, cdm_shrink_z = cdm_all[3]
        dis_pot_dens, dis_pot_shrink = cdm_all[4], cdm_all[5]
        # cdm_fof,cdm_pot,cdm_dens,cdm_shrink,dis_pot_dens,dis_pot_shrink
        # name = names[f]
        prop_num_save[N_ok] = np.array((R_delta, N_delta,
                                        cdm_fof_x, cdm_fof_y, cdm_fof_z,
                                        cdm_pot_x, cdm_pot_y, cdm_pot_z,
                                        cdm_dens_x, cdm_dens_y, cdm_dens_z,
                                        cdm_shrink_x, cdm_shrink_y, cdm_shrink_z,
                                        cdm_cdm_x, cdm_cdm_y, cdm_cdm_z))
        prop_full_save = prop_full_save + [[halo_name_f, R_delta, N_delta,
                                           cdm_fof_x, cdm_fof_y, cdm_fof_z,
                                           cdm_pot_x, cdm_pot_y, cdm_pot_z,
                                           cdm_dens_x, cdm_dens_y, cdm_dens_z,
                                           cdm_shrink_x, cdm_shrink_y, cdm_shrink_z,
                                           cdm_cdm_x, cdm_cdm_y, cdm_cdm_z]]
        name_save = name_save + [halo_name_f]
        R_and_M = np.append(R_and_M, N_FOF_all[f])
        print(R_and_M)
        R_and_M_save[N_ok] = R_and_M
        N_ok += 1  # incrementation
        # if N_ok == N_ok_threshold :
        #    break

print('N_ok =', N_ok)

# R_delta_all = pd.DataFrame(R_delta_all,columns=['Rvir (Mpc/h)'])
# R_delta_all.to_csv(path_out+'halo_properties/Rvir_all_'+str(n_parallel)+'.dat',index=False)
# N_delta_all = pd.DataFrame(N_delta_all,columns=['Nvir'])
# N_delta_all.to_csv(path_out+'halo_properties/Nvir_all_'+str(n_parallel)+'.dat',index=False)
if N_ok > 0:
    prop_num_save = prop_num_save[0:N_ok]
    prop_num_save = pd.DataFrame(prop_num_save, columns=['Rvir (Mpc/h)', 'Nvir',
                                                         'cdm_fof_x (Mpc/h)', 'cdm_fof_y', 'cdm_fof_z',
                                                         'cdm_pot_x (Mpc/h)', 'cdm_pot_y', 'cdm_pot_z',
                                                         'cdm_dens_x (Mpc/h)', 'cdm_dens_y', 'cdm_dens_z',
                                                         'cdm_shrink_x (Mpc/h)', 'cdm_shrink_y', 'cdm_shrink_z',
                                                         'cdm_x (Mpc/h)', 'cdm_y', 'cdm_z'])
    R_and_M_save = R_and_M_save[0:N_ok]
    R_and_M_save = pd.DataFrame(R_and_M_save, columns=['Rvir (Mpc/h)', 'Nvir', 'R200m', 'N200m', 'R200c', 'N200c',
                                                       'R500c', 'N500c', 'R1000c', 'N1000c', 'R2000c', 'N2000c', 'NFOF'])
    name_save = pd.DataFrame(name_save, columns=['halo name'])
    prop_full_save = pd.DataFrame(prop_full_save, columns=['halo name', 'Rvir (Mpc/h)', 'Nvir',
                                                           'cdm_fof_x (Mpc/h)', 'cdm_fof_y', 'cdm_fof_z',
                                                           'cdm_pot_x (Mpc/h)', 'cdm_pot_y', 'cdm_pot_z',
                                                           'cdm_dens_x (Mpc/h)', 'cdm_dens_y', 'cdm_dens_z',
                                                           'cdm_shrink_x (Mpc/h)', 'cdm_shrink_y', 'cdm_shrink_z',
                                                           'cdm_x (Mpc/h)', 'cdm_y', 'cdm_z'])
    if save_prop == True:
        prop_num_save.to_csv(
            path_out+'halo_properties/prop_saved_'+str(n_parallel)+'.dat', index=False)
        R_and_M_save.to_csv(
            path_out+'halo_properties/R_and_M_saved_'+str(n_parallel)+'.dat', index=False)
        name_save.to_csv(path_out+'halo_properties/names_saved_' +
                         str(n_parallel)+'.dat', index=False)
        prop_full_save.to_csv(
            path_out+'halo_properties/prop_names_saved_'+str(n_parallel)+'.dat', index=False)
t_fin = time.time()
print('total time =', t_fin - t_in)
