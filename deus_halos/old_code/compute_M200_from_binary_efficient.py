#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:39:34 2020

@author: guillaume

This code looks for halos in a given mass range coming from the FOF algorithm of DEUS.
It computes their mass, number of particles, center of mass, highest density point, 
virial radius and shift.
"""

import struct
import numpy as np
import pandas as pd
import unsiotools.simulations.cfalcon as falcon
cf=falcon.CFalcon()
import sys
import time


save_prop, save_data = False, False
step_log = 0.2
cluster = False
if cluster == True :
    n = np.int(sys.argv[1]) # I import a number from a terminal command line
    mass_log = np.int(sys.argv[2]) + 0.5 # mass log 
    if np.int(sys.argv[3]) == 0 :
        cosmo = 'rpcdm'
    elif np.int(sys.argv[3]) == 1 :
        cosmo = 'wcdm'
else :
    n = 1
    mass_log = 12.5
    cosmo = 'wcdm'

file_param_name = 'parameters_DEUS_fof_boxlen648_n2048_'+cosmo+'w7'
#from 'parameters_DEUS_fof_boxlen648_n2048_'+cosmo+'w7' import hsml, L_box, rho_vir_Brian_Norman, mass_one_particle   
file_param = __import__(file_param_name) 
hsml = file_param.hsml
L_box = file_param.L_box
rho_vir_Brian_Norman = file_param.rho_vir_Brian_Norman
mass_one_particle = file_param.mass_one_particle


print(hsml, L_box, rho_vir_Brian_Norman, mass_one_particle)

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
    output_number = np.array((R_delta,N_delta,error_0,error_fof))
    output_array = np.array((r_sorted,ind,radius,rho))
    return(output_number, output_array)
    
def compute_cdm_with_phi(x,y,z,ind_phi,N_mean) :
    ''' 
    Compute the cdm from a list of phi function indices (potential gravitational or local density)
    The center is the center of mass around the particle with its N_mean neighbours
    '''
    cdm_x_0 = x[ind_phi[0]] # potential minimum point
    cdm_y_0 = y[ind_phi[0]]
    cdm_z_0 = z[ind_phi[0]]
    radius = np.sqrt((x-cdm_x_0)**2 + (y-cdm_y_0)**2 + (z-cdm_z_0)**2)
    ind_radius_sorted = np.argsort(radius)
    cdm_x = np.mean(x[ind_radius_sorted[0:N_mean]])
    cdm_y = np.mean(y[ind_radius_sorted[0:N_mean]])
    cdm_z = np.mean(z[ind_radius_sorted[0:N_mean]])
    cdm_phi = np.array((cdm_x,cdm_y,cdm_z)) # halo center in Mpc/h
    return(cdm_phi)

def cdm_shrinked_sphere(x,y,z,fac_N=0.01,fac_rad=0.025,r_max=-1,N_min=100) :
    ''' It computes the shrinking sphere center.
    It returns it in the frame of x, y and z and in the same unit
    '''
    # Initialisation
    N_part_initial = len(x)
    cdm = np.array((np.mean(x),np.mean(y),np.mean(z)))
    x_i = x - cdm[0]
    y_i = y - cdm[1]
    z_i = z - cdm[2]
    radius_i = np.sqrt(x_i**2 + y_i**2 + z_i**2)
    if r_max == -1 :
        r_max = np.max(radius_i)
    N_part_i = len(x)
    fac_N = 0.01
    i = 0
    while (N_part_i > fac_N * N_part_initial and N_part_i > N_min) : # loop
        r_threshold_i = r_max * (1 - fac_rad)**i
        ind_i = np.where(radius_i < r_threshold_i)[0]
        x_i, y_i, z_i = x_i[ind_i], y_i[ind_i], z_i[ind_i]
        cdm_i = np.array((np.mean(x_i),np.mean(y_i),np.mean(z_i)))
        x_i_use = x_i - cdm_i[0]
        y_i_use = y_i - cdm_i[1]
        z_i_use = z_i - cdm_i[2]
        radius_i = np.sqrt(x_i_use**2 + y_i_use**2 + z_i_use**2)
        N_part_i = len(ind_i)
        i += 1
    cdm_shrink = cdm_i + cdm
    return(cdm_shrink)
        
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
    cdm_fof = np.array((np.mean(x),np.mean(y),np.mean(z))) # standard center of mass in Mpc/h
    # I reset the particle position in this approximate center frame
    x_use, y_use, z_use = x - cdm_fof[0], y - cdm_fof[1], z - cdm_fof[2]
    N_part = len(x_use) # number of particles
    # I reset the particle positions in order to give it as input for Dehnen
    pos = np.zeros((N_part,3),dtype=np.float32) 
    pos[:,0], pos[:,1], pos[:,2] = x_use, y_use, z_use
    pos = np.reshape(pos,N_part*3)
    mass_array = np.ones((N_part),dtype=np.float32)
    ok, acc, pot = cf.getGravity(pos,mass_array,hsml)
    # with the minimum of the potential gravitational, I compute the halo center: cdm_pot
    ind_pot = np.argsort(pot)
    cdm_pot_use = compute_cdm_with_phi(x_use,y_use,z_use,ind_pot,N_mean)
    cdm_pot = cdm_pot_use + cdm_fof
    # with the local density, I compute the highest density point: cdm_dens
    ok, rho_one_particle, h_sml = cf.getDensity(pos,mass_array)
    ind_rho = np.argsort(rho_one_particle)
    ind_rho = ind_rho[::-1] # reverse the order: put the densest first
    cdm_dens_use = compute_cdm_with_phi(x_use,y_use,z_use,ind_rho,N_mean)
    cdm_dens = cdm_dens_use + cdm_fof
    # shrinking sphere
    cdm_shrink_use = cdm_shrinked_sphere(x_use,y_use,z_use)
    cdm_shrink = cdm_shrink_use + cdm_fof
    # distances between pot and dens in Mpc/h
    dis_pot_dens = np.sqrt( (cdm_pot[0]-cdm_dens[0])**2 + (cdm_pot[1]-cdm_dens[1])**2 + (cdm_pot[2]-cdm_dens[2])**2 )
    dis_pot_shrink = np.sqrt( (cdm_pot[0]-cdm_shrink[0])**2 + (cdm_pot[1]-cdm_shrink[1])**2 + (cdm_pot[2]-cdm_shrink[2])**2 )
    return(cdm_fof,cdm_pot,cdm_dens,cdm_shrink,dis_pot_dens,dis_pot_shrink)

def periodicity(x,y,z,cdm,dis_1=0.5,dis_2=0.9,box=1):
    ''' This function checks if there is a periodicity problem with the data,
    because particles could lie at the border of the simulation box.
    So for e.g. a particle p1 in such halo could have a position x[p1]=0.01
    and another particle p2 in the same halo x[p2]=0.99 In the simulation 
    those 2 particles are considered at a distance of ~0.02 (periodicity condition)
    x, y, z contains the particle positions in box unit.
    cdm is the halo center in box unit, chosen as the position of any particle, the 0th
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

def read_binary_file(path_in,path_out,name_file,f,Mass_halo_min,Mass_halo_max,
                     save_prop,save_data,
                     N_mean=32,L_box=L_box,
                     rho_delta=rho_vir_Brian_Norman,factor_mass_marge=10,
                     N_part_threshold=200):
    '''This functions takes the output of the DEUS simulation of kind comoving 
    and halo particles for e.g 
    http://www.deus-consortium.org/deus-data/snapshots-data/#files
    The file should be located at path_in, its name should be name_file, 
    its numero should be f. It saves the data of halos of a given mass M_delta 
    between Mass_halo_min and Mass_halo_maximum in particle mass unit, inside 
    new files located in path_out. It checks that the halo position are not 
    at the border of the simulation box. If it is the case, it reorganizes 
    the positions. It computes R_delta, where delta is by default
    Brian and Norman:  https://iopscience.iop.org/article/10.1086/305262/pdf
    The center of mass in a sphere (of 1 Mpc/h by default) is computed 
    as well as the highest density point, which is used, because thought more reliable. 
    The highest density point is computed with the Dehnen function. 
    By default, the default argument of the Dehnen function are used,
    but it can be modified with Dehnen_argument, see
    https://pypi.org/project/python-unsiotools/
    and by writing: pydoc unsiotools.simulations.cfalcon
    In the file, we only consider the halo in a mass_fof range
    [Mass_halo_min/factor_mass_marge,Mass_halo_max*factor_mass_marge]
    because mass_fof and M_delta are different. 
    factor_mass_marge is by default set to 10.
    N_part_threshold is the minimu number of particles where I consider that a halo is a halo
    N_mean is the number of particles that I consider to compute 
    the center of mass in the densest area (see the functtion compute_centers)
    This function returns the followings:
    N_part_in_halos_fof: contains the particle number of all halos (FOF).
    N_part_in_halos_delta: contains the particle number of all halos 
    in the mass range [Mhalomin,Mhalomax] (M_delta).
    error_file_0 : contains file name where the density threshold is too low.
    error_file_fof : contains file name where the density threshold is too high.
    name_halo_usefull : contains all names of halos in the mass range wanted.
    properties : contains the name, the particle number (FOF), the particle number (delta)
    the center of mass, the highest density point, the shift in Mpc/h unit.
    prop : contains the particle number (FOF), the particle number (delta)
    the center of mass, the highest density point, the shift in Mpc/h unit.
    '''
    # I set the name of the halo with problem in error_file_0 and error_file_fof
    error_file_0, error_file_fof = [], []
    error_file_0_real = []
    error_0_tot, error_fof_tot = 0, 0
    name_halo_usefull = [] # names of the halos with the M_delta tha I want
    properties, prop, = [], [] # properties of the halos with the M_delta that I want
    file = open(path_in+name_file+'_'+str('{:d}'.format(f).zfill(5))+"", "rb") # I open the binary file number f                                                                                         
    file.read(4) # number bites                                                                                                        
    N_halos = struct.unpack('<i', file.read(4))[0] # number of halos in the file f
    print('N_halos =',N_halos)     
    # N_part_fof in all halos of the file f and N_vir in all halos that I want
    N_part_in_halos_fof, N_part_in_halos_delta = np.zeros((N_halos),dtype=int), np.zeros((N_halos,2),dtype=int)     
    N_begin, N_end = 0, N_halos
    h_d = 0 # number of halo with the mass M_delta that I want in the file f
    for h in range(N_begin,N_end): #I do a loop on the number of halo in the file f  
        file.read(8) # number bites
        N_part = struct.unpack('<i', file.read(4))[0] # number of particles in the halo h of the file f  
        N_part_in_halos_fof[h] = N_part
        file.read(8) # number bites
        # I compute the properties of halo with a mass between a minimu and a maximum mass with a marge factor
        if Mass_halo_min/factor_mass_marge < N_part < Mass_halo_max*factor_mass_marge and N_part > N_part_threshold :  
            # position of particles from the FOF in unit of boxlength
            pos = struct.unpack('<'+str(N_part*3)+'f', file.read(4*3*N_part))
            # I reshape the position, and set it in np.float64, in order to avoid doublons
            pos = np.array(pos,dtype=np.float64).reshape(N_part,3).transpose()
            # look for https://stackoverflow.com/questions/16165488/packing-and-unpacking-binary-float-in-python
            # and http://www.deus-consortium.org/a-propos/cosmological-models/post-processing/, 
            # it tells me that it is only simple precision, so np.float32
            x, y, z = pos[0], pos[1], pos[2]
            cdm_rough = np.array((x[0], y[0], z[0])) # be carefull, I can not compute the cdm, even roughly at this step.
            # I need to check the periodicity before
            # I check the periodicity
            x_new, y_new, z_new, bad_periodicity = periodicity(x,y,z,cdm_rough)
            # I compute Rvir, Nvir, the halo center, the center of mass
            x, y, z = x_new * L_box, y_new * L_box, z_new * L_box
            #cdm_pot, cdm_dens, cdm_fof, dis_pot_dens = compute_centers(x, y, z, N_mean)
            cdm_fof,cdm_pot,cdm_dens,cdm_shrink,dis_pot_dens,dis_pot_shrink = compute_centers(x, y, z, N_mean)
            out_num, out_array = compute_R_delta(cdm_pot,x,y,z,rho_delta)   
            R_delta, N_delta, error_0, error_fof = out_num[0], out_num[1], out_num[2], out_num[3]
            error_0_tot, error_fof_tot = error_0_tot + error_0, error_fof_tot + error_fof
            # I compute the shift in Mpc/h unit
            shift = np.sqrt( (cdm_fof[0]-cdm_pot[0])**2 + (cdm_fof[1]-cdm_pot[1])**2 + (cdm_fof[2]-cdm_pot[2])**2 )
            if error_0 != 0 :
                # If error_0 is non empty, then there was a mistake or a problem
                error_file_0 = error_file_0 + [[''+name_file+'_'+str('{:d}'.format(f).zfill(5))+'_'+str('{:d}'.format(h).zfill(5))+'',N_part,N_delta,error_0]]
            if error_fof != 0 :
                # In error_file_fof, I put the names of haloes from which I could not computed properly Mvir, and I need to go beyond the FOF to do it              
                error_file_fof = error_file_fof + [[''+name_file+'_'+str('{:d}'.format(f).zfill(5))+'_'+str('{:d}'.format(h).zfill(5))+'',N_part,N_delta,error_fof,cdm_fof[0],cdm_fof[1],cdm_fof[2]]]
            if Mass_halo_min < N_delta < Mass_halo_max :
                # case where the M_vir of the halo is what I wanted, 
                # I will save it with its velocities, identities and properties
                N_part_in_halos_delta[h_d] = np.array((h,N_delta))
                file.read(8)
                velocity = struct.unpack('<'+str(N_part*3)+'f', file.read(4*3*N_part))
                velocity = np.array(velocity,dtype=np.float64) # in float64, such as the positions
                file.read(8)
                identity = struct.unpack('<'+str(N_part)+'q', file.read(8*N_part))
                identity = np.array(identity,dtype=np.int64) # in int64, as indicated on the deus-consortium
                # I reshape
                pos = np.array((x_new,y_new,z_new)).transpose().reshape(1,3*N_part)[0]
                data = np.append(np.append(pos,velocity),identity)
                #data_save = pd.DataFrame(data,columns=['x (Mpc/h)','y (Mpc/h)','z (Mpc/h)','v_x (DEUS unit)','v_y (DEUS unit)','v_z (DEUS unit)','id'])
                #data_save.to_csv(path_out+'data/'+name_file+'_'+str('{:d}'.format(f).zfill(5))+'_'+str('{:d}'.format(h).zfill(5))+'.dat')
                if save_data == True :
                    np.savetxt(path_out+'data/'+name_file+'_'+str('{:d}'.format(f).zfill(5))+'_'+str('{:d}'.format(h).zfill(5))+'.dat',data)
                name_halo_usefull = name_halo_usefull + [''+name_file+'_'+str('{:d}'.format(f).zfill(5))+'_'+str('{:d}'.format(h).zfill(5))+'.dat']
                properties = properties + [[''+name_file+'_'+str('{:d}'.format(f).zfill(5))+'_'+str('{:d}'.format(h).zfill(5))+'.dat',
                                            N_part,np.int(N_delta),R_delta,
                                            cdm_fof[0],cdm_fof[1],cdm_fof[2],
                                            cdm_pot[0],cdm_pot[1],cdm_pot[2],
                                            cdm_dens[0],cdm_dens[1],cdm_dens[2],
                                            cdm_shrink[0],cdm_shrink[1],cdm_shrink[2],
                                            shift,dis_pot_dens,dis_pot_shrink]]
                prop = prop + [[N_part,np.int(N_delta),R_delta,cdm_fof[0],cdm_fof[1],cdm_fof[2],
                                cdm_pot[0],cdm_pot[1],cdm_pot[2],cdm_dens[0],cdm_dens[1],cdm_dens[2],
                                cdm_shrink[0],cdm_shrink[1],cdm_shrink[2],shift,dis_pot_dens,dis_pot_shrink]]
                h_d += 1
            else :
                # case where the halo have an interesting M_fof (according to factor_mass_marge)
                # but its exact M_vir is not what I wanted
                # and so I do not save it at the end
                number = 8 + 12*N_part + 8 + 8*N_part
                file.read(number)
        else :
            # case where the M_fof of the halo is very different 
            # (according to factor_mass_marge) of what I wanted
            # and so I do not consider it
            number = 12*N_part + 8 + 12*N_part + 8 + 8*N_part
            file.read(number)    
    if error_0_tot > 0 :
        # if there were error_0 (0 particle inside the density threshold), 
        error_file_0_save = pd.DataFrame(error_file_0,columns=['halo name','N_part_FOF','N_part_vir','error (0 = good): you have particles with higher density than delta_vir'])
        if save_prop == True :
            error_file_0_save.to_csv(path_out+'Problem/errors_0_in_'+name_file+'_'+str('{:d}'.format(f).zfill(5))+'.dat',index=False)
    if error_fof_tot > 0 :
        # if there were error_fof (N_part_tot inside the density threshold)
        error_fof_save = pd.DataFrame(error_file_fof,columns=['halo name','N_part_FOF','N_part_vir','error (0 = good): you have particles with lower density than delta_vir ','cdm_x (Mpc/h)','cdm_y','cdm_z'])
        if save_prop == True :
            error_fof_save.to_csv(path_out+'Problem/errors_fof_in_'+name_file+'_'+str('{:d}'.format(f).zfill(5))+'.dat',index=False)
    N_part_in_halos_delta = N_part_in_halos_delta[0:h_d]  
    print('Number of haloes with the good mass M_delta: h_d=',h_d)
    return(N_part_in_halos_fof,N_part_in_halos_delta,name_halo_usefull,properties,prop)
      
def apply_main(f_begin,f_end,path_in,path_out,name_file,Mass_min,Mass_max,save_prop=False,save_data=False):
    name_halo_usefull = []
    properties = []
    prop = []
    for f in range(f_begin,f_end): 
        # MAIN COMPUTATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print('file f =',f)
        results = read_binary_file(path_in,path_out,name_file,f,Mass_min,Mass_max,save_prop,save_data)
        # I SAVE what is important, the Npart and Nvir of all the haloes, 
        # the name and the properties of the haloes with the Mvir that I wanted
        # I save the number of particles in each binary file           
        #N_FOF_save = pd.DataFrame(results[0],columns=['N_part_FOF'])
        #N_FOF_save.to_csv(path_out+'N_part/N_part_fof_in_all_haloes_'+name_file+'_'+str('{:d}'.format(f).zfill(5))+'.dat')
        #N_vir_save = pd.DataFrame(results[1],columns=['halo num','N_part_vir'])
        #N_vir_save.to_csv(path_out+'N_part/N_vir_in_all_haloes_'+name_file+'_'+str('{:d}'.format(f).zfill(5))+'.dat')
        name_halo_usefull = name_halo_usefull + results[2]
        properties = properties + results[3]
        prop = prop + results[4]
    name_halo_usefull_save = pd.DataFrame(name_halo_usefull,columns=['halo usefull name'])
    #print(properties)
    properties_save = pd.DataFrame(properties,columns=['halo name','N_part_FOF','N_part_vir','Rvir',
                                                       'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                       'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                       'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                                       'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                                       'shift s= |r_cpot-r_cfof|/Rvir','|r_cpot-r_cdens|','|r_cpot-r_cshrink|'])
    prop_save = pd.DataFrame(prop,columns=['N_part_FOF','N_part_vir','Rvir',
                                           'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                           'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                           'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                           'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                           'shift s= |r_cpot -r_cfof|/Rvir','|r_cpot-r_cdens|','|r_cpot-r_cshrink|'])
    if save_prop == True :
        name_halo_usefull_save.to_csv(path_out+'halo_properties/haloes_usefull_'+str(n)+'.dat',index=False)
        properties_save.to_csv(path_out+'halo_properties/properties_'+str(n)+'.dat',index=False)
        prop_save.to_csv(path_out+'halo_properties/properties_numbers_'+str(n)+'.dat',index=False)
    return()

def indice_cluster(n):
    # if I want to apply to the cluster
    N_haloes = 2876 # Number of haloes
    N_cores = 128 # I am using N_cores + 1 cores: 32*16=512
    multiple = 4 # I cut my 2881 haloes in multiples times 2880/multiples = N_cores to parallelize on 2881/multiples + 1 cores
    if n <= N_cores :
        f_begin = np.int(multiple*(n-1)) # begining and end of my loop
        f_end = np.int(multiple*n)
    else :
        f_begin = multiple*N_cores
        f_end = N_haloes
    return(f_begin,f_end)


#n=1, 31, 33, 519, 88 it's better, you need to look for 8 cubes
f_begin, f_end = indice_cluster(n)
f_begin, f_end = 0, 2
print(f_begin,f_end)
# Parameters
#f_begin, f_end = 0, 3 #512
name_file = 'fof_boxlen648_n2048_'+cosmo+'w7_strct' # name and path of my binary data
#path_in = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/Data_binary/' # for my laptop
#path_in = '../../../../DEUSS_648Mpc_2048_particles/Data_binary/' #(for cluster cc.lam)  
#path_in = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+str(mass_log)+'_Msun_FOF/halo_particles/'
path_in = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/halo_particles/'
#path_out = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_'+str(mass_log)+'_Msun_FOF/' # for my laptop
path_out = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+str(mass_log)+'_Msun_FOF/' # for cluster cc.lam 
marge_M_delta_min = 1.1
marge_M_delta_max = 1.1
Mass_min = ((10**(mass_log))/mass_one_particle)/marge_M_delta_min # virial mass minimum that I want for my halo in Msun/h
Mass_max = marge_M_delta_max*(10**(mass_log + step_log))/mass_one_particle # virial mass maximum that I want for my halo in Msun/h                          

t_begin = time.time()
apply_main(f_begin,f_end,path_in,path_out,name_file,Mass_min,Mass_max,save_prop=save_prop,save_data=save_data)
t_end = time.time()
print(t_end-t_begin)

