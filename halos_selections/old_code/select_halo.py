#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 08:05:34 2021

@author: guillaume

Here I want to select a halo in a sphere among a file .dat from DEUS 
containing halo and its neighbourhood (a box of size 2*5*Rvir)
"""





import numpy as np
import matplotlib.pyplot as plt
import random
import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()

#import sys
#path = '../' # path to the code I want to import
#sys.path.append(''+path+'')
from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import L_box, hsml, G_Newton, v_ren, rho_crit, rho_vir_Brian_Norman


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

def select_sphere(path_prop,path_data,path_save,N_halos,fac):
    #path = '../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/'
    #path_save = '../../../DEUSS_648Mpc_2048_particles/output_not_MF/sampling_effect/halo_simu_selection/'
    names = np.loadtxt(''+path_prop+'names_sel_random.dat',dtype=str)
    prop = np.loadtxt(''+path_prop+'prop_sel_random.dat')
    for h in range(N_halos) :
        name = names[h]
        print(name)
        name = name.replace('.dat','',1) # remove once the string .dat in the string, it respects the sorting, so it does not remove any a e.g., if present
        data = np.loadtxt(''+path_data+''+name+'.dat_5_Rvir_new.dat')
        data = data.transpose()
        N_part_tot = len(data)
        print(N_part_tot)
        x, y, z = data[:,0], data[:,1], data[:,2]
        cdm = prop[h,5:]
        Rvir = prop[h,0]
        Nvir = prop[h,1]
        print(cdm)
        print(Rvir)
        print(Nvir)
        x, y, z, bad_periodicity = periodicity(x,y,z,cdm/L_box)
        print(bad_periodicity)
        x, y, z = x * L_box, y * L_box, z * L_box
        radius = np.sqrt( (x-cdm[0])**2 + (y-cdm[1])**2 + (z-cdm[2])**2 )
        ind_r = np.argsort(radius)
        r_sorted = radius[ind_r]
        ind_sel = np.where(r_sorted < fac * Rvir)[0]
        ind_sel_sphere = ind_r[ind_sel]
        data = data[ind_sel_sphere]
        pos = data[:,0:3] * L_box - cdm
        print(len(pos))
        #np.savetxt(''+path_save+''+name+'_sel_sphere_'+str(fac)+'_Rvir.dat',pos)
    return()


def multi_scatter_plot(pos,N_rows=5,N_col=6) :
    fig, ax = plt.subplots(nrows=N_rows,ncols=N_col,figsize=[10,10])
    row, col = 0, 0
    lt, lb, ll, lr = False, False, False, False
    for h in range(N_halos) :
        x_MF, y_MF, z_MF = pos[h,:,0], pos[h,:,1], pos[h,:,2]
        if row == 0 :
            lt = True
        else :
            lt = False
        if row == N_rows - 1 :
            lb = True
        else :
            lb = False
        if col == 0 :
            ll = True
        else :
            ll = False
        if col == N_col - 1 :
            lr = True
        else :
            lr = False
        print(h,row,col)
        ax[row,col].tick_params(which='both', length=5, width=1, direction="in", 
                                top=lt, bottom=lb, left=ll, right=lr,
                                labeltop=lt,labelbottom=lb, labelleft=ll, labelright=lr)
        ax[row,col].scatter(x_MF,z_MF,s=0.1,c='b')
        col += 1
        if col == N_col :
            col = 0
            row += 1
            if row == N_rows :
                row -= 1
                col = N_col - 1
                break
        fig.text(0.48,0.05,'$x$ $(Mpc/h)$')
        fig.text(0.05,0.48,'$z$ $(Mpc/h)$',rotation=90)
    plt.savefig(''+path_save+'DEUS_30_halos_mass_10_14_5_xz_axis.pdf',format='pdf',transparent=True)
    plt.show()
    return()

def data_for_MF(pos, Rvir, my_push) :
    pos = pos + Rvir + my_push * Rvir
    return(pos)

def sampling_sphere(path_prop,path_data,path_save,fac,N_halos,N_times,N_part_MF,my_push,dec=2) :
    #path = '../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/'
    #path_save = '../../../DEUSS_648Mpc_2048_particles/output_not_MF/sampling_effect/halo_simu_selection/'
    names = np.loadtxt(''+path_prop+'names_sel_random.dat',dtype=str)
    prop = np.loadtxt(''+path_prop+'prop_sel_random.dat')    
    shift = np.loadtxt(''+path_prop+'cdm_and_shift_sel_random.dat')[:,3]
    shape = np.loadtxt(''+path_prop+'abc_sphericity_CV_sel_random.dat')
    q = np.loadtxt(''+path_prop+'equilibrium_sel_random.dat')[:,1]
    S, Q, T = shape[:,3], shape[:,4], shape[:,5]
    #N_rows, N_col = 5, 6
    #fig, ax = plt.subplots(nrows=N_rows,ncols=N_col,figsize=[10,10])
    #row, col = 0, 0
    #lt, lb, ll, lr = False, False, False, False
    pos_MF = np.zeros((N_halos,N_part_MF,3))
    for h in range(N_halos) :
        name = names[h]
        name = name.replace('.dat','',1)
        Rvir = prop[h,0] # virial radius in Mpc/h
        print(h)
        print('name = ',name)
        print('Rvir = ',Rvir)
        print('shift = ',shift[h])
        print('shape : S = ',S[h],' Q = ',Q[h],'T = ',T[h])
        print('virial_ratio = ',q[h])
        pos = np.loadtxt(''+path_data+''+name+'_sel_sphere_'+str(fac)+'_Rvir.dat') # position in the highest density point of the halo, in Mpc/h
        pos = data_for_MF(pos, Rvir, my_push) # my_push should be dimensionless
        #print(pos)
        N_part = len(pos)
        for n in range(N_times) :
            #print(n)
            my_integers = random.sample(range(N_part),N_part_MF) # sampling
            pos_MF[h] = pos[my_integers] # in r_s unit
            #np.savetxt(''+path_save+''+name+'_sel_sphere_'+str(fac)+'_Rvir_Npart_'+str(N_part_MF)+'_'+str(n)+'_large_radius.dat',pos_MF[h])
    multi_scatter_plot(pos_MF)
    return()


def sampling_iso(fac=1,step=0.1,N_halos=3,N_part_MF=2000,my_push=0.4) :
    path = '../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/'
    path_save = '../../../DEUSS_648Mpc_2048_particles/output_not_MF/sampling_effect/halo_simu_selection/'
    names = np.loadtxt(''+path+'halo_properties/names_all.dat',dtype=str)
    prop = np.loadtxt(''+path+'halo_properties/prop_all.dat')    
    shift = np.loadtxt(''+path+'halo_properties/cdm_and_shift.dat')[:,3]
    shape = np.loadtxt(''+path+'halo_properties/cluster_3/abc_sphericity_CV_all.dat')
    S, Q, T = shape[:,3], shape[:,4], shape[:,5]
    for h in range(2,N_halos) :
        name = names[h]
        name = name.replace('.dat','',1)
        print('name = ',name)
        print('shift = ',shift[h])
        print('shape : S = ',S[h],' Q = ',Q[h],'T = ',T[h])
        pos = np.loadtxt(''+path_save+''+name+'_sel_sphere_'+str(fac)+'_Rvir.dat')
        Rvir = prop[h,0]
        print('Rvir = ', Rvir)
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        radius = np.sqrt( x**2 + y**2 + z**2 ) # Mpc/h
        ind_r = np.argsort(radius)
        pos = pos[ind_r]
        r_sorted = radius[ind_r] # Mpc/h
        N_part = len(pos)
        pos = np.array(np.reshape(pos,3*N_part),dtype=np.float32) # Mpc/h
        mass_array = np.ones((N_part),dtype=np.float32)
        ok, acc, potential = cf.getGravity(pos, mass_array, hsml, G=1.0, theta=0.6, kernel_type=1, ncrit=6)
        pot = np.abs(potential)  
        ind_moins = np.where(r_sorted < (fac) * Rvir)[0]
        ind_shell = np.where(r_sorted[ind_moins] > (fac -step) * Rvir)[0]
        pot_shell = pot[ind_moins][ind_shell]
        pot_threshold = np.mean(pot_shell)
        ind_pot = np.where(pot > pot_threshold)[0]
        N_part_pot = len(ind_pot)
        print(pos)
        pos = np.array(np.reshape(pos,(N_part,3)))
        print(pos)
        pos_pot = pos[ind_pot]
        print(N_part_pot)
        my_integers = random.sample(range(N_part_pot),N_part_MF) # sampling
        pos_pot = data_for_MF(pos_pot, Rvir, my_push) 
        pos_MF = pos_pot[my_integers] # in r_s unit
        x, y, z = pos_pot[:,0], pos_pot[:,1], pos_pot[:,2]
        x_MF, y_MF, z_MF = pos_MF[:,0], pos_MF[:,1], pos_MF[:,2]
        plt.scatter(x,z,s=0.001,c='b')
        plt.scatter(x_MF,z_MF,s=1,c='r')
        plt.show()
        np.savetxt(''+path_save+''+name+'_sel_iso_'+str(fac)+'_Rvir_MF.dat',pos_MF)
    return()

if __name__ == '__main__' :
    #select_sphere()
    path_prop = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/halo_properties/cluster_3/'
    path_data = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/'
    path_save = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/limit_case/'
    N_halos = 30
    fac = 1
    #N_part_MF = 1000
    #my_push_array = [0.6,0.5,0.4,0.3,0.2]
    N_times_array = [30,30,30,30,3]
    N_part_MF_array = [100,300,1000,3000,10000]
    #n = 0
    my_push = 10
    '''for n in range(5) :
        N_times = N_times_array[n]
        N_part_MF = N_part_MF_array[n]
        #my_push = my_push_array[n]
        print(' \n N_part_MF = ',N_part_MF)
        print('my_push = ',my_push)
        print('N_times = ',N_times)
        sampling_sphere(path_prop,path_data,path_save,fac,N_halos,N_times,N_part_MF,my_push)
        #break
        #n += 1'''
    #select_sphere(path_prop,path_data,path_save,N_halos,fac)
    sampling_sphere(path_prop,path_data,path_save,fac,N_halos,N_times=1,N_part_MF=1000,my_push=0.4)
    #sampling_iso()