#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:11:24 2021

@author: guillaume
"""


'''
This script can be used to compute the shift
'''

import numpy as np
from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import L_box, hsml, G_Newton, v_ren
import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()
import pandas as pd


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

def compute_cdm(f_beg,f_end,path,names,add_name,cdm_dens,Rvir):

    #N_haloes = 100
    N_haloes = f_end - f_beg
    cdm_x, cdm_y, cdm_z = np.zeros((N_haloes)), np.zeros((N_haloes)), np.zeros((N_haloes))
    #path = '../input_MF/data_halo/isopotential_selection/mass_10_12.5_Mvir/cluster/pos_full/'
    for h in range(f_beg,f_end):
        print(h)
        my_h = h - f_beg
        name = names[h]
        name_file = ''+name+''+add_name+''
        pos = np.loadtxt(''+path+''+name_file+'')
        x, y, z = pos[0], pos[1], pos[2]
        print(x)
        x, y, z, bad_periodicity = periodicity(x,y,z,cdm_dens[my_h]/L_box)
        x, y, z = x*L_box, y*L_box, z*L_box
        print(x)
        cdm_x[my_h], cdm_y[my_h], cdm_z[my_h] = np.mean(x), np.mean(y), np.mean(z)
    shift = (np.sqrt( (cdm_x-cdm_dens[:,0])**2 + (cdm_y-cdm_dens[:,1])**2 + (cdm_z-cdm_dens[:,2])**2 ))/Rvir
    print(shift)
    return(cdm_x,cdm_y,cdm_z,shift)

def compute_centers(x, y, z, Dehnen_argument, N_mean=32, R_delta=10):
    ''' This function computes the halo center by computing the center of mass inside
    the densest area of the halo, containg N_mean particles.
    It also computes the basic center of mass in a sphere of a threshold radius R_delta
    R_delta is 10 Mpc/h by default, in order to get the FOF center of mass
    Dehnen_default_parameter can be found here:
    https://pypi.org/project/python-unsiotools/
    and by writing: pydoc unsiotools.simulations.cfalcon
    '''
    # Dehnen default arguments
    N_nearest_neighbours = np.int(Dehnen_argument[0])
    N_ferrer = np.int(Dehnen_argument[1])
    method = np.int(Dehnen_argument[2])
    #ncrit = np.int(Dehnen_argument[3])
    N_part = len(x) # number of particles
    # I reset the particle positions in order to give it as input for Dehnen
    pos = np.zeros((N_part,3),dtype=np.float32) 
    pos[:,0], pos[:,1], pos[:,2] = x, y, z
    pos = np.reshape(pos,N_part*3)
    mass_array = np.ones((N_part),dtype=np.float32)
    cf=falcon.CFalcon()
    ok,rho_one_particle,hsml = cf.getDensity(pos,mass_array, N_nearest_neighbours,
                                             N_ferrer,method)#,ncrit)
    # with the local density, I compute the halo center
    ind_rho_one_particle_sorted = np.argsort(rho_one_particle)
    cdm_x_0 = x[ind_rho_one_particle_sorted[-1]]
    cdm_y_0 = y[ind_rho_one_particle_sorted[-1]]
    cdm_z_0 = z[ind_rho_one_particle_sorted[-1]]
    radius = np.sqrt((x-cdm_x_0)**2 + (y-cdm_y_0)**2 + (z-cdm_z_0)**2)
    ind_radius_sorted = np.argsort(radius)
    cdm_x = np.mean(x[ind_radius_sorted[0:N_mean]])
    cdm_y = np.mean(y[ind_radius_sorted[0:N_mean]])
    cdm_z = np.mean(z[ind_radius_sorted[0:N_mean]])
    cdm_dens = np.array((cdm_x,cdm_y,cdm_z)) # halo center in Mpc/h
    radius = np.sqrt((x-cdm_x)**2 + (y-cdm_y)**2 + (z-cdm_z)**2)
    ind_sphere = np.where(radius < R_delta)[0]    
    x_sphere, y_sphere, z_sphere = x[ind_sphere], y[ind_sphere], z[ind_sphere]
    cdm = np.array((np.mean(x_sphere),np.mean(y_sphere),np.mean(z_sphere))) # center of mass in Mpc/h
    return(cdm_dens,cdm)
       
def compute_q_approx(pos,K_tot,N_part):
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
    # set pos in Mpc/h and vel in km/s
    mass_array = np.ones((N_part),dtype=np.float32)
    ok, acc, phi = cf.getGravity(pos, mass_array, hsml, 
                                       G=1.0, theta=0.6, kernel_type=1, ncrit=6)
    # I reset the potential gravitational energy in physical unit
    V_all = phi * G_Newton
    # I compute the potential energy of my halo in (Megameter/S)**2
    V_tot = np.sum(V_all)*0.5 # facteur 1/2 formule
    q = 2*K_tot/np.abs(V_tot)
    return(q)

def selection_sphere(pos,Rvir) :
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    rad = np.sqrt(x**2 + y**2 + z**2)
    ind_sphere = np.where(rad < Rvir)[0]
    pos_sphere = pos[ind_sphere]
    return(pos_sphere,ind_sphere)
    

if __name__ == '__main__' :
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_14_5_sphere_sel/properties/'
    path_haloes = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/sphere_sel_1_Rvir/'
    prop = np.loadtxt(path+'prop_all.dat')
    names_all = np.genfromtxt(path+'names_all.dat',dtype=str)
    Rvir = prop[:,0]
    cdm_fof_x, cdm_fof_y, cdm_fof_z = prop[:,2], prop[:,3], prop[:,4]
    cdm_dens_x, cdm_dens_y, cdm_dens_z = prop[:,5], prop[:,6], prop[:,7]
    shift = np.sqrt( (cdm_fof_x-cdm_dens_x)**2 + (cdm_fof_y-cdm_dens_y)**2 + (cdm_fof_z-cdm_dens_z)**2 )
    shift /= Rvir
    r = 0
    N_haloes = len(shift)
    for h in range(N_haloes) :
        if shift[h] < 0.07:
            r += 1
            
    print(r)
    #print(names_all)
    #names = 'fof_boxlen648_n2048_lcdmw7_strct_00002_02733_sphere_sel_1_Rvir.dat'
    Dehnen_argument = [2, 1, 0, None]
    N_haloes = 2 # 328
    shift_new = np.zeros((N_haloes))
    names_non_shifted_haloes = []
    shift_non_shifted_haloes = []
    indices_non_shifted_haloes = []
    r = 0
    for h in range(N_haloes) :
        #names = 'fof_boxlen648_n2048_lcdmw7_strct_00002_02733_sphere_sel_1_Rvir.dat'
        name = names_all[h]
        name_used = name.replace('.dat', '_sphere_sel_1_Rvir.dat')
        print(name_used)
        pos = np.loadtxt(path_haloes+name_used)
        #N_vir = len(pos)
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        cdm_dens,cdm = compute_centers(x, y, z, Dehnen_argument, N_mean=32, R_delta=Rvir[0])
        shift_new[h] = (np.sqrt( (cdm_dens[0]-cdm[0])**2 + (cdm_dens[1]-cdm[1])**2  + (cdm_dens[2]-cdm[2])**2))/Rvir[h]
        #K_tot = kinetic(vel)
        #compute_q(pos,K_tot,N_vir)
        
        if shift_new[h] < 0.07 : # Neto+07 criteria
            r += 1
            names_non_shifted_haloes += [name]
            shift_non_shifted_haloes += [shift_new[h]]
            indices_non_shifted_haloes += [h]
        
    #print(shift_new)
    shift_save = pd.DataFrame(shift_new,columns=['shift: s=|cdm_dens-cdm_sphere|/Rvir'])
    #print(shift_save)
    #print(names_non_shifted_haloes)
    #print(r)
    #np.savetxt(path+'names_non_shifted_haloes.dat',names_non_shifted_haloes,fmt='%s')
    #np.savetxt(path+'shift_non_shifted_haloes.dat',shift_non_shifted_haloes)
    #np.savetxt(path+'indices_non_shifted_haloes.dat',indices_non_shifted_haloes)
    #shift_save.to_csv(path+'shift_cdm_dens_cdm_sphere_Rvir_all.dat')
    
    
    
    '''radius = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    print(np.min(radius),np.max(radius))
    print(Rvir[0])
    cdm_sphere_x = np.mean(pos[:,0])
    print(cdm_sphere_x)
    pos_dehnen = np.reshape(pos,3*N_vir)
    pos_dehnen = np.array(pos_dehnen,dtype=np.float32)
    mass_dehnen = np.ones((N_vir),dtype=np.float32)
    ok, rho_loc, hsml_array = cf.getDensity(pos_dehnen, mass_dehnen, K=32, N=1, method=0, ncrit=None)
    ind_rho = np.argsort(rho_loc)
    pos_sorted = pos[ind_rho]
    cdm_x_dens, cdm_y_dens, cdm_z_dens = '''
    
    
    
    
