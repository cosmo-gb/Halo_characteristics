#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 08:59:55 2022

@author: guillaume
"""



'''
This script computes ellipsoidal parameters of halos a, b and c
which are the three main axis length of the ellipsoid for the DEUS halo at different redshift and cosmology.
It also computes the passage matrice, allowing to rotate the halo, thus leading to
x <=> a, y <=> b and z <=> c.
It has been tested by generating triaxial halos with axis in any directions,
not aligned with x,y,z and it works well (typically <~ 1% error for N_part >~ 10^4).
But be carefull, I think that when I go more in the halo center region, 
I need more particles to be accurate.
I clarely need to investigate more the validity of this code at r <~ 0.1 Rvir.
'''


import numpy as np
import time
import pandas as pd
import sys

cluster = False
if cluster == True : # case where I apply this code on the cluster
    # for the cluster array run:
    #0,1,2 <=> lcdm, mass=12,13,14
    #3,4,5 <=> rpcdm, mass=12,13,14
    #6,7,8 <=> wcdm, mass=12,13,14
    num_cosmo_mass = np.int(sys.argv[1])
    if num_cosmo_mass < 3 :
        cosmo = 'lcdm'
    elif num_cosmo_mass < 6 :
        cosmo = 'rpcdm'
    else :
        cosmo = 'wcdm'
    reste = num_cosmo_mass % 3
    if reste == 0 :
        mass_str_num = '12_5'
    elif reste == 1 :
        mass_str_num = '13_5'
    else :
        mass_str_num = '14_5'
    z_str = '0'
    N_haloes = 2
    save_file = True
    do_complete = True
else : # case where I apply this code on my laptop
    cosmo = 'rpcdm'
    z_str = '0'
    mass_str_num = '12_5'
    N_haloes = 2
    save_file = True
    do_complete = False
    
mass_str = 'mass_10_'+mass_str_num+'_Msun_beyond_FOF'       
file_param_name = 'param_DEUS_box_648_'+cosmo+'w7_z_'+z_str
file_param = __import__(file_param_name) 

L_box = file_param.L_box
rho_vir_Brian_Norman = file_param.rho_vir_Brian_Norman
mass_one_particle = file_param.mass_one_particle
rho_crit_0 = file_param.rho_crit_0
Omega_matter_0 = file_param.Omega_matter_0
redshift = file_param.redshift
v_ren = file_param.v_ren
hsml = file_param.hsml


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

def put_in_frame(pos,Rvir,Nvir,cdm,box):
    x, y, z = pos[:,0]/box, pos[:,1]/box, pos[:,2]/box
    x, y, z, bad_periodicity = periodicity(x,y,z,cdm/box)
    pos_frame = np.zeros((Nvir,3))
    pos_frame[:,0], pos_frame[:,1], pos_frame[:,2] = x*box, y*box, z*box
    pos_frame -= cdm
    pos_frame /= Rvir
    return(pos_frame)

def put_in_frame_lcdm_mass(pos,Rvir):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    rad = np.sqrt(x**2 + y**2 + z**2)
    if np.max(rad) > 1.001*Rvir :
        print('What !!!! There is a periodicity problem')
        print('maximal radius in virial unit =',np.max(rad)/Rvir)
        sys.exit('I need to solmve a periodicity issue !')
    pos_frame = pos/Rvir
    return(pos_frame)

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

def do_computation_test(path,name,N_samples=1,unit=1,r_max=1) :
    # pos can be originally in some other unit than unit
    # r_max should be in unit unit,
    # e.g. pos can be in Mpc/h, unit can Rvir in Mpc/h and r_max can be directly in Rvir
    # the sphericity computation can be tested with this function
    # you have typically 5%, 2% and 0.5% of precision for haloes with 10**3, 10**4 and 10**5 particles respectively
    abc = np.zeros((N_samples,3))
    for s in range(N_samples) :
        print(s)
        pos = np.loadtxt(path+name+str(s)+'.dat')
        print(pos)
        theta_x, theta_y, theta_z = 5, 108, 17
        pos = rotate(pos,theta_x,theta_y,theta_z)
        print(pos)
        #r_max = 0.05 # in the same unit as pos # 0.02 (10**6), 0.06(10**5) 0.09 (10**4)
        pos /= unit # I reset in Rvir unit
        #r_max /= unit
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        r = np.sqrt(x**2 + y**2 + z**2)
        my_max = np.max(r)
        print(my_max)
        ind = np.where(r < r_max)[0]
        x, y, z, r = x[ind], y[ind], z[ind], r[ind]
        my_max_2 = np.max(r)
        if my_max != my_max_2 :
            print('r_max not equal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
            print('r_max_sel =',my_max_2,'r_max_before=',my_max)
            print('r_max_th=',r_max,unit,'N_part_sel=',len(r))
        # a, b, c, S, Q, T, Passage, error, initial number of iteration
        SQT_Passage = compute_ellipsoid_parameters(x,y,z,a_unit=1/r_max)
        abc[s] = SQT_Passage[0:3]
        print('a, b, c = ',SQT_Passage[0:3])
        SQT, Passage = reorganise(SQT_Passage)
    abc_mean = np.mean(abc,axis=0)
    abc_std = np.std(abc,axis=0)
    print(abc_mean)
    print(abc_std)
    print(abc_std/abc_mean)
    return()

########################################################################################
# if you want to test the code
#path = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/cosmo_DE_z_impact/test/SA_halos/'
#name = 'halo_Npart_1000_abg_131_c_5_abc_10705_'
#N_samples = 2
#do_computation_test(path,name,N_samples=N_samples)
#########################################################################################

if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+z_str+'/'+cosmo+'/'+mass_str+'/'
    names_all = np.array(pd.read_csv(path+'halo_properties/names_sel.dat'))[:,0]
    if do_complete == True :
        N_haloes = len(names_all)
    prop_all = np.array(pd.read_csv(path+'halo_properties/properties_numbers_sel.dat'))
    Rvir_all = prop_all[:,0]
    Nvir_all = np.array(prop_all[:,1],dtype=int)
    cdm_all = prop_all[:,5:8]
    print('L_box =',L_box)
    SQT, Passage = np.zeros((N_haloes,9)), np.zeros((N_haloes,9))
    for h in range(N_haloes) :
        halo_name = names_all[h].replace('.dat','_sphere_1_Rvir.dat')
        print(h,halo_name)
        data =  np.array(pd.read_csv(path+'data/sphere_1_Rvir/'+halo_name,sep=',', comment='#'))
        pos = data[:,0:3]
        Rvir = Rvir_all[h]
        Nvir = Nvir_all[h]
        cdm = cdm_all[h]
        if cosmo != 'lcdm' or mass_str == 'most_massive' :
            pos_frame = put_in_frame(pos,Rvir,Nvir,cdm,L_box)
        else :
            pos_frame = put_in_frame_lcdm_mass(pos,Rvir)
        x, y, z = pos_frame[:,0], pos_frame[:,1], pos_frame[:,2]
        SQT_Passage = compute_ellipsoid_parameters(x,y,z)
        print('a, b, c = ',SQT_Passage[0:3])
        SQT[h], Passage[h] = reorganise(SQT_Passage)
        
    file_save = open(path+'halo_properties/abc_SQT.dat', 'w')
    file_save.write('# Sphericity parameters \n')
    file_save.write('# L_box=648Mpc/h, '+cosmo+'w7, z = '+z_str+' '+mass_str+' haloes, with Brian&Norman98 selection \n')
    file_save.write('# a, b, c, S, Q, T, error, iteration, frac_ell  \n')
    np.savetxt(file_save,SQT)
    file_save.close()
    
    file_save_p = open(path+'halo_properties/Passage.dat', 'w')
    file_save_p.write('# Passage matrix \n')
    file_save_p.write('# L_box=648Mpc/h, '+cosmo+'w7, z = '+z_str+' '+mass_str+' haloes, with Brian&Norman98 selection \n')
    file_save_p.write('# P11, P12, P13, P21, P22, P23, P31, P32, P33  \n')
    np.savetxt(file_save_p,Passage)
    file_save_p.close()

