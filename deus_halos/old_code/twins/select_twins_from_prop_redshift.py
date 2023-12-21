#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:51:31 2022

@author: guillaume
"""

'''
This code finds the twins halos among a given list of halos of different redshifts
'''

import numpy as np
import pandas as pd
import sys
from random import sample


cluster = False
if cluster == True : # case where I apply this code on the cluster
    cosmo = str(sys.argv[1])
    mass_str = str(sys.argv[2])
    N_halos = 256
    frac_box = 0.2
    save_file = True
else : # case where I apply this code on my laptop
    frac_box = 0.2
    cosmo = 'rpcdm'
    mass_str = 'most_massive'
    N_halos = 256
    save_file = False
    

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

def deal_with_cdm(cdm_ref,cdm_test,frac=0.1,L_box=1) :
    ind_x = np.where( (np.abs(cdm_ref[0] - cdm_test[:,0]) < frac*L_box) # either very close
                         |(np.abs(cdm_ref[0] - cdm_test[:,0]) > ( 1-frac)*L_box))[0] # or very far
    ind_y = np.where( (np.abs(cdm_ref[1] - cdm_test[ind_x,1]) < frac*L_box) 
                         |(np.abs(cdm_ref[1] - cdm_test[ind_x,1]) > ( 1-frac)*L_box))[0]
    ind_z = np.where( (np.abs(cdm_ref[2] - cdm_test[ind_x[ind_y],2]) < frac*L_box) 
                         |(np.abs(cdm_ref[2] - cdm_test[ind_x[ind_y],2]) > ( 1-frac)*L_box))[0]
    ind_found = ind_x[ind_y[ind_z]]
    N_found = len(ind_found)
    #ind[0:N_found] = ind_found
    cdm_x, cdm_y, cdm_z, pb = periodicity(cdm_test[ind_found,0],cdm_test[ind_found,1],cdm_test[ind_found,2],cdm_ref)
    #print(cdm_save_pb)
    #cdm_save = np.array([cdm_save_pb[0][0], cdm_save_pb[1][0], cdm_save_pb[2][0]])
    #print(cdm_save)
    #print(cdm_save[0])
    return(ind_found, N_found, cdm_x, cdm_y, cdm_z)

def find_good_candidate(cdm_all, N_halos, frac, L_box=1, N_red=5) :
    ind_good = np.ones((N_red,N_halos,N_halos),dtype=int) * (-1)
    ind_good[0,:,0] = np.array(range(N_halos),dtype=int)
    #ind_3, ind_4 = np.ones((N_halos,N_halos),dtype=int) * (-1), np.ones((N_halos,N_halos),dtype=int) * (-1)
    cdm_save = np.zeros((N_red,N_halos,N_halos,3))
    cdm_save[0,:,0] = cdm_all[0,0:N_halos]
    N_found_good = np.zeros((N_red,N_halos),dtype=int) #, np.zeros((N_halos),dtype=int)
    N_found_good[0,:] = 1
    #N_found_3, N_found_4 = np.zeros((N_halos),dtype=int), np.zeros((N_halos),dtype=int)
    for h in range(N_halos) :
        cdm_ref = cdm_all[0,h]
        for z in range(1,N_red) :
            ind_found_h, N_found_h, cdm_x_h, cdm_y_h, cdm_z_h = deal_with_cdm(cdm_ref,cdm_all[z],frac=frac,L_box=L_box)
            N_found_good[z,h] = N_found_h
            ind_good[z,h,0:N_found_good[z,h]] = ind_found_h
            cdm_save[z,h,0:N_found_good[z,h],0] = cdm_x_h
            cdm_save[z,h,0:N_found_good[z,h],1] = cdm_y_h
            cdm_save[z,h,0:N_found_good[z,h],2] = cdm_z_h
    return(ind_good,N_found_good,cdm_save)


def sort_cdm_by_distance(cdm_save,N_found,N_halos,N_red=5) :
    ind_sort = np.ones((N_red,N_halos,N_halos),dtype=int)*(-1) #, np.ones((N_halos,N_halos),dtype=int)*(-1)
    ind_sort[0,:,0] = 0
    for h in range(N_halos) :
        cdm_ref = cdm_save[0,h,0]
        for z in range(1,N_red) :
            cdm_z = cdm_save[z,h,0:N_found[z,h]]
            dis_z = np.sqrt( (cdm_ref[0]-cdm_z[:,0])**2 + (cdm_ref[1]-cdm_z[:,1])**2 + (cdm_ref[2]-cdm_z[:,2])**2 )
            ind_sort[z,h,0:N_found[z,h]] = np.argsort(dis_z)
        #cdm_1 = cdm_save[1,h,0:N_found_1[h]]
        #dis_1 = np.sqrt( (cdm_ref[0]-cdm_1[:,0])**2 + (cdm_ref[1]-cdm_1[:,1])**2 + (cdm_ref[2]-cdm_1[:,2])**2 )
        #ind_1[h,0:N_found_1[h]] = np.argsort(dis_1)
        ####################################################################################"
        #cdm_2 = cdm_save[2,h,0:N_found_2[h]]
        #dis_2 = np.sqrt( (cdm_ref[0]-cdm_2[:,0])**2 + (cdm_ref[1]-cdm_2[:,1])**2 + (cdm_ref[2]-cdm_2[:,2])**2 )
        #ind_2[h,0:N_found_2[h]] = np.argsort(dis_2)
    return(ind_sort)

def check_id_particles(id_1,id_2,N_s_1=10) :
    N_1, N_2 = len(id_1), len(id_2)
    ind_s_1 = sample(range(N_1),N_s_1)
    id_s_1 = id_1[ind_s_1]
    i_1 = 0
    for i in id_s_1 :
        if len(np.where(id_2 == i)[0]) == 1 :
            i_1 += 1
    return(i_1,i_1/N_s_1)

def look_for_twins(names_all,ind,N_found,ind_sort,N_halos,N_red=5,N_sample=100,frac_th=0.5) :
    path_0 = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/'+mass_str+'/'
    i_save = np.ones((N_red,N_halos),dtype=int) * (-1)
    i_save[0] = np.array(range(N_halos),dtype=int)
    frac_save = np.zeros((N_red,N_halos))
    frac_save[0] = 1
    for h in range(N_halos) :
        halo_name = names_all[0,h]+'_sphere_1_Rvir.dat'
        print('LCDM, halo =',h,halo_name)
        data_0 =  np.array(pd.read_csv(path_0+'data/'+halo_name))
        id_0 = np.array(data_0[:,6],dtype=int)
        #########################################################################"
        z = 1
        for z_str in ['0_5','1','1_5','2'] :
            path_z = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
            print('Number of interesting halos for z='+z_str+' is ',N_found[z,h])
            names_z_h = names_all[z,ind[z,h,ind_sort[z,h,0:N_found[z,h]]]]
            for i in range(N_found[z,h]) :
                halo_name = names_z_h[i]+'_sphere_1_Rvir.dat'
                print('z = '+z_str+' halo =',h,halo_name)
                data_z =  np.array(pd.read_csv(path_z+'data/'+halo_name))
                id_z = np.array(data_z[:,6],dtype=int)
                N_id_z, frac_id_z = check_id_particles(id_z,id_0,N_sample)
                print('good id ratio =', frac_id_z)
                if frac_id_z > frac_th :
                    i_save[z,h] = ind[z,h,ind_sort[z,h,0:N_found[z,h]]][i]
                    frac_save[z,h] = frac_id_z
                    break
            z += 1
    return(i_save,frac_save)
                
            

if __name__ == '__main__':
    path_0 = '../../../../DEUSS_648Mpc_2048_particles/'
    N_red = 5
    N_sample = 1000
    L_box = np.zeros((N_red))
    cdm_all = np.zeros((N_red,N_halos,3))
    c = 0
    names_all = np.zeros((N_red,N_halos),dtype=object)
    z_str_array = ['0','0_5','1','1_5','2']
    for z_str in z_str_array :
        file_param_name = 'param_DEUS_box_648_'+cosmo+'w7_z_'+z_str
        file_param = __import__(file_param_name) 
        L_box[c] = file_param.L_box
        path_end = '/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
        path_MF_out = path_0+'output_MF/cosmo_DE_z_impact'+path_end
        path_prop = path_0+'mass_bin_data'+path_end+'halo_properties/'
        names_all[c] = np.array(pd.read_csv(path_prop+'names_sel.dat'))[:,0]
        prop_all = np.array(pd.read_csv(path_prop+'properties_numbers_sel.dat'))
        cdm_all[c] = prop_all[:,5:8]/L_box[c]
        c += 1
    ind, N_found, cdm_save = find_good_candidate(cdm_all,N_halos,frac=frac_box,N_red=N_red)
    print('Number of no good centers for z=0.5 is ',len(np.where(N_found[1] == 0)[0]))
    print('Number of no good centers for z=1 is ',len(np.where(N_found[2] == 0)[0]))
    print('Number of no good centers for z=1.5 is ',len(np.where(N_found[3] == 0)[0]))
    print('Number of no good centers for z=2 is ',len(np.where(N_found[4] == 0)[0]))
    ind_sort = sort_cdm_by_distance(cdm_save,N_found,N_halos,N_red=N_red)
    i_save, frac_save = look_for_twins(names_all,ind,N_found,ind_sort,1,N_red=N_red,N_sample=N_sample)
    f_save = np.zeros((N_red,1,2))
    f_save[:,:,0] = i_save
    f_save[:,:,1] = frac_save
    print(i_save, frac_save)
    z = 1
    if save_file == True :
        for z_str in z_str_array[1:] :
            path_end = '/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
            path_prop = path_0+'mass_bin_data'+path_end+'halo_properties/'
            file_save = open(path_prop+'twins_halo_redshift.dat', 'w')
            file_save.write('# identity of the halos wich are the twins (if it exists) of the z=0 halos \n')
            file_save.write('# L_box=648Mpc/h, '+cosmo+'w7, z = '+z_str+' '+mass_str+' halos, with Brian&Norman98 selection \n')
            file_save.write('# halo identity, identical particles ratio \n')
            np.savetxt(file_save,f_save[z])
            file_save.close()
            z += 1
    