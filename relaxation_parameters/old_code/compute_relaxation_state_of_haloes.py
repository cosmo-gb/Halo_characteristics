#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:41:41 2021

@author: guillaume
"""

"""
This script computes different notions of centers for FOF data and data inside a sphere of Rvir.
The centers are the standard mass center (cdm), the highest density point (cdm_dens), 
the gravitational potential minimum (cdm_pot) and the shrinking sphere (cdm_shrink).
It also computes the shift and the virial ratio q=2K/|V|
Then it creates list of relaxed halos according to different definitions.
see https://ui.adsabs.harvard.edu/abs/1996ApJ...467..489C/abstract
or Neto+07
"""


import numpy as np
import pandas as pd
import unsiotools.simulations.cfalcon as falcon
cf=falcon.CFalcon()

from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import hsml, v_ren, L_box, rho_vir_Brian_Norman, mass_one_particle, G_Newton

def periodicity(x,y,z,cdm,dis_1=0.5,dis_2=0.9,box=1):
    ''' This function checks if there is a periodicity problem with the data,
    because particles could lie at the border of the simulation box.
    So for e.g. a particle p1 in such halo could have a position x[p1]=0.01
    and another particle p2 in the same halo x[p2]=0.99 In the simulation 
    those 2 particles are considered at a distance of ~0.02 (periodicity condition)
    x, y, z contain the particle positions in box unit.
    cdm is the halo center in box unit (any notion of center is fine).
    dis_1 is the threshold distance allowed from the center
    so if a particles is at a smaller distance than dis_1 from the center, 
    then I consider that there is no problem, by default dis_1 = 0.5
    dis_2 is the distance in one direction, which allows me to test 
    if the particle p is close to 0 or 1, by default dis_2 = 0.9
    box is the length of the box simulation, by default box = 1
    It returns the particle positions without periodicity problem
    and bad_periodicity=0 if there was no problem or bad_periodicity=1
    if the problem had been removed.
    The particle positions ordering is unchanged.
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

def compute_q_approx(pos,K_tot,N_part):
    ''' It computes the virial ratio approximatively
    by using an approximation on the gravitational field (spherical symmetry)
    pos are the particle positions (Mpc/h)
    K_tot are the total kinetic energy
    N_part is the number of particles
    '''
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
    ''' This function computes the virial ratio 
    pos is the particle position in Mpc/h 
    it should be a float32 array of 3*N_part dimension (Dehnen function)
    vel is the particle velocities in km/s
    N_part is the number of particle
    '''
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
        
def get_gravity_one_particle(x,y,z,ind) :
    x_p, y_p, z_p = x[ind], y[ind], z[ind]
    x, y, z = x-x_p, y-y_p, z-z_p
    radius = np.sqrt(x**2 + y**2 + z**2)
    ind_r = np.argsort(radius)
    r_not_p = radius[ind_r][1:]
    my_phi = -1/r_not_p
    phi = np.sum(my_phi)
    return(phi)

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
    '''
    #cdm_fof = np.array((0,0,0))
    cdm_fof = np.array((np.mean(x),np.mean(y),np.mean(z))) # standard center of mass in Mpc/h
    N_part = len(x) # number of particles
    # I reset the particle positions in order to give it as input for Dehnen
    x_use, y_use, z_use = x - cdm_fof[0], y - cdm_fof[1], z - cdm_fof[2]
    print('x_use[0] =',x_use[0])
    r = np.sqrt(x_use**2 + y_use**2+ z_use**2)
    print('r_min=',np.min(r),'r_max =',np.max(r))
    pos = np.zeros((N_part,3),dtype=np.float32) 
    pos[:,0], pos[:,1], pos[:,2] = x_use, y_use, z_use
    pos = np.reshape(pos,N_part*3)
    mass_array = np.ones((N_part),dtype=np.float32)
    ok, acc, pot = cf.getGravity(pos,mass_array,hsml)
    pot_min = np.min(pot)
    print('pot_min =',pot_min)
    ind_pot_min = np.where(pot == pot_min)[0]
    print(ind_pot_min)
    my_phi = get_gravity_one_particle(x_use,y_use,z_use,ind_pot_min)
    print('my_phi_min',my_phi)
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
    dis_pot_dens = np.sqrt( (cdm_pot[0]-cdm_dens[0])**2 + (cdm_pot[1]-cdm_dens[1])**2 + (cdm_pot[2]-cdm_dens[2])**2 )
    #cdm_fof = np.array((np.mean(x),np.mean(y),np.mean(z))) # standard center of mass in Mpc/h
    return(cdm_pot,cdm_dens,cdm_fof,cdm_shrink,dis_pot_dens)

def compare_cdm(cdm_1,cdm_2,R_unit=1) :
    dis = np.sqrt( (cdm_1[0]-cdm_2[0])**2 + (cdm_1[1]-cdm_2[1])**2 + (cdm_1[2]-cdm_2[2])**2 )
    dis /= R_unit
    return(dis)

def compute_virial_ratio(vel,pos,N_part) :
    vel_mean = np.mean(vel,axis=0)
    vel -= vel_mean
    vel *= v_ren
    K_tot = kinetic(vel)
    pos_Dehnen = np.reshape(pos,N_part*3)
    pos_Dehnen = np.array(pos_Dehnen,dtype=np.float32)
    q = compute_q(pos_Dehnen,K_tot,N_part)
    q_approx = compute_q_approx(pos,K_tot,N_part)
    return(q,q_approx)

def check_relaxed(cdm_pot_fof,cdm_pot_sphere,cdm_dens_fof,cdm_dens_sphere,
                  cdm_shrink_fof,cdm_shrink_sphere,cdm_sphere,cdm_fof) :
    shift_Neto = np.sqrt(cdm_sphere[0]**2 + cdm_sphere[1]**2 + cdm_sphere[2]**2)/Rvir
    print('shift Neto+07 =',shift_Neto)
    shift_me = np.sqrt(cdm_fof[0]**2 + cdm_fof[1]**2 + cdm_fof[2]**2)/Rvir
    print('shift me =',shift_me)
    rel_Neto = 0 # if it stays 0, it means it is relaxed
    rel_me = 0
    if shift_Neto > 0.07 :
        rel_Neto = 1
    if shift_me > 0.07 :
        rel_me = 1
    
    # virial ratio
    q_fof, q_approx_fof = compute_virial_ratio(vel_fof,pos_fof,N_fof)
    if q_fof > 1.35 :
        rel_me = 1
    q_sphere, q_approx_sphere = compute_virial_ratio(vel_sphere,pos_sphere,Nvir)
    if q_sphere > 1.35 :
        rel_Neto = 1
    print('q_Neto+07 =',q_sphere)
    print('q_me =',q_fof)
    # test for relaxed halos   
    cdms = np.array((cdm_pot_fof,cdm_pot_sphere,cdm_dens_fof,cdm_dens_sphere,
                     cdm_shrink_fof,cdm_shrink_sphere))
    cdms_2 = cdms**2
    dis = np.sqrt(np.sum(cdms_2,axis=1))
    ind_bad_cdm = np.where(dis > 3*hsml)[0]
    if len(ind_bad_cdm) != 0 :
        rel_me = 1
    cdms_fof = np.array((cdm_pot_fof,cdm_dens_fof,cdm_shrink_fof))
    cdms_sphere = np.array((cdm_pot_sphere,cdm_dens_sphere,cdm_shrink_sphere))
    dis_diff = cdms_fof - cdms_sphere
    dis_diff = dis_diff**2
    dis_diff = np.sum(dis_diff,axis=1)
    dis_diff = np.sqrt(dis_diff)
    ind_pb = np.where(dis_diff > hsml)[0]
    if len(ind_pb) != 0 :
        rel_me = 1
    print('rel_me =',rel_me)
    print('rel_Neto =',rel_Neto)
    return(rel_me,rel_Neto,shift_me,shift_Neto,q_fof,q_sphere)

cluster = False
mass_log = '13.5'

if cluster == True :
    path_use = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/'
    path_data = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/data/'
    path_prop = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/halo_properties/'
    path_fof = path_use+'/mass_10_'+mass_log+'_Msun_FOF/data/'
    path_out = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/'
else :
    path_use = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data'
    path_data = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/data/'
    path_prop = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/halo_properties/cluster_4/'
    path_fof = path_use+'/mass_10_'+mass_log+'_Msun_FOF/data/'
    path_out = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/'

names_all = np.array(pd.read_csv(path_prop+'names_all.dat'))[:,0]
prop_all = np.array(pd.read_csv(path_prop+'properties_numbers_all.dat'))
cdm_pot_fof = np.array(( prop_all[:,5], prop_all[:,6], prop_all[:,7])).transpose()
#cdm_fof = np.array((prop_all[:,2], prop_all[:,3], prop_all[:,4])).transpose()
#cdm_dens_fof = np.array((prop_all[:,8], prop_all[:,9], prop_all[:,10])).transpose()
Rvir_array = prop_all[:,0]
Nvir_array = prop_all[:,1]

ind_rel_me = []
ind_rel_Neto = []
#N_halos = len(Rvir_array)
N_halos = 1
print('N_halos =',N_halos)
prop_fof = np.zeros((N_halos,14))
prop_sphere = np.zeros((N_halos,14))
relaxation = np.zeros((N_halos,4))

for h in range(N_halos) : #0,N_halos) :
    # I set all the positions in the frame of the cdm_pot_fof
    # here I compute the centers in the FOF
    name_file_fof = names_all[h]
    print(name_file_fof)
    data_fof = np.loadtxt(path_fof+name_file_fof)
    N_fof = len(data_fof)//7
    print('N_part_fof =',N_fof)
    print(cdm_pot_fof[h])
    pos_fof = data_fof[:3*N_fof].reshape((N_fof,3))
    vel_fof = data_fof[3*N_fof:6*N_fof].reshape((N_fof,3))
    pos_fof *= L_box
    #pos_fof -= cdm_pot_fof[h]
    x_fof, y_fof, z_fof = pos_fof[:,0], pos_fof[:,1], pos_fof[:,2]
    cdm_pot_fof_new,cdm_dens_fof,cdm_fof,cdm_shrink_fof,dis_pot_dens = compute_centers(x_fof, y_fof, z_fof)
    print('cdm_pot_fof =',cdm_pot_fof_new) # + cdm_pot_fof[h])
    #print('cdm_dens_fof =',cdm_dens_fof + cdm_pot_fof[h])
    #print('cdm_fof =',cdm_fof + cdm_pot_fof[h])
    #print('cdm_shrink_fof =',cdm_shrink_fof + cdm_pot_fof[h])
    break
    
    # here I load the data beyond the FOF and check the periodicity
    name_file = names_all[h].replace('.dat','.dat_5_Rvir_new.dat')
    print(name_file)
    data = np.array(pd.read_csv(path_data+name_file))
    N_part_tot = len(data)
    x, y, z = data[:,0], data[:,1], data[:,2]
    v_x, v_y, v_z = data[:,3], data[:,4], data[:,5]
    vel = np.array((v_x,v_y,v_z)).transpose()
    Rvir = Rvir_array[h]
    Nvir = np.int(Nvir_array[h])
    x,y,z,bad_periodicity = periodicity(x,y,z,cdm_pot_fof[h]/L_box)
    print('bad periodicity =',bad_periodicity)
    
    # here I compute the centers inside a sphere of 1 Rvir
    # again I reset everything in the frame of the potential center computed from the FOF data
    pos = np.array((x,y,z)).transpose()
    pos *= L_box
    pos -= cdm_pot_fof[h]
    radius = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    ind_sphere = np.where(radius < Rvir)[0]
    pos_sphere = pos[ind_sphere]
    x_sphere, y_sphere, z_sphere = pos_sphere[:,0], pos_sphere[:,1], pos_sphere[:,2]
    vel_sphere = vel[ind_sphere]
    print('Nvir =',Nvir)
    cdm_pot_sphere,cdm_dens_sphere,cdm_sphere,cdm_shrink_sphere,dis_pot_dens = compute_centers(x_sphere, y_sphere, z_sphere)
    print('cdm_pot_sphere =',cdm_pot_sphere)
    print('cdm_dens_sphere =',cdm_dens_sphere)
    print('cdm_sphere =',cdm_sphere)
    print('cdm_shrink_sphere =',cdm_shrink_sphere)
    
    rel_me, rel_Neto, shift_me, shift_Neto, q_fof, q_sphere = check_relaxed(cdm_pot_fof_new,cdm_pot_sphere,cdm_dens_fof,cdm_dens_sphere,
                                                                         cdm_shrink_fof,cdm_shrink_sphere,cdm_sphere,cdm_fof)
    
    relaxation_use = np.array((shift_me, shift_Neto, q_fof, q_sphere))
    relaxation[h] = relaxation_use
    
    if rel_me == 0 :
        ind_rel_me += [h]
    if rel_Neto == 0 :
        ind_rel_Neto += [h]
    
    cdm_fof += cdm_pot_fof[h]
    cdm_pot_fof_new += cdm_pot_fof[h]
    cdm_dens_fof += cdm_pot_fof[h]
    cdm_shrink_fof += cdm_pot_fof[h]
    
    prop_fof_use = np.array((Rvir,N_fof,
                             cdm_fof[0],cdm_fof[1],cdm_fof[2],
                             cdm_pot_fof_new[0],cdm_pot_fof_new[1],cdm_pot_fof_new[2],
                             cdm_dens_fof[0],cdm_dens_fof[1],cdm_dens_fof[2],
                             cdm_shrink_fof[0],cdm_shrink_fof[1],cdm_shrink_fof[2]))
    prop_fof[h] = prop_fof_use
    
    cdm_sphere += cdm_pot_fof[h]
    cdm_pot_sphere += cdm_pot_fof[h]
    cdm_dens_sphere += cdm_pot_fof[h]
    cdm_shrink_sphere += cdm_pot_fof[h]
    
    prop_sphere_use = np.array((Rvir,Nvir,
                                cdm_sphere[0],cdm_sphere[1],cdm_sphere[2],
                                cdm_pot_sphere[0],cdm_pot_sphere[1],cdm_pot_sphere[2],
                                cdm_dens_sphere[0],cdm_dens_sphere[1],cdm_dens_sphere[2],
                                cdm_shrink_sphere[0],cdm_shrink_sphere[1],cdm_shrink_sphere[2]))
    prop_sphere[h] = prop_sphere_use
    
    
prop_fof = pd.DataFrame(prop_fof,columns=['Rvir (Mpc/h)','N_fof',
                                          'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                          'cdm_pot_fof_x (Mpc/h)','cdm_pot_fof_y','cdm_pot_fof_z',
                                          'cdm_dens_fof_x (Mpc/h)','cdm_dens_fof_y','cdm_dens_fof_z',
                                          'cdm_shrink_fof_x (Mpc/h)','cdm_shrink_fof_y','cdm_shrink_fof_z'])
#prop_fof.to_csv(path_out+'halo_properties/cdm_all_fof.dat',index=False)

prop_sphere = pd.DataFrame(prop_sphere,columns=['Rvir (Mpc/h)','Nvir',
                                                'cdm_sphere_x (Mpc/h)','cdm_sphere_y','cdm_sphere_z',
                                                'cdm_pot_sphere_x (Mpc/h)','cdm_pot_sphere_y','cdm_pot_sphere_z',
                                                'cdm_dens_sphere_x (Mpc/h)','cdm_dens_sphere_y','cdm_dens_sphere_z',
                                                'cdm_shrink_sphere_x (Mpc/h)','cdm_shrink_sphere_y','cdm_shrink_sphere_z'])
#prop_sphere.to_csv(path_out+'halo_properties/cdm_all_sphere.dat',index=False)

relaxation = pd.DataFrame(relaxation,columns=['s=|r_cpotfof - r_cfof|/Rvir',
                                              's=|r_cpotfof - r_cspherevir|/Rvir',
                                              'q=2K/|V| (fof)','q=2K/|V| (sphere Rvir)'])
#relaxation.to_csv(path_out+'halo_properties/relaxation.dat',index=False)

ind_rel_me = np.array(ind_rel_me,dtype=int)
ind_rel_me = pd.DataFrame(ind_rel_me,columns=['indices of relaxed halos with Mvir = [10^14.5,10^14.7] Msun/h (shift_fof<0.07, q_fof<1.35 and coherent centers)'])
#ind_rel_me.to_csv(path_out+'halo_properties/relaxed_halos_me.dat',index=False)

ind_rel_Neto = np.array(ind_rel_Neto,dtype=int)
ind_rel_Neto = pd.DataFrame(ind_rel_Neto,columns=['indices of relaxed halos with Mvir = [10^14.5,10^14.7] Msun/h (Neto+07: shift_vir<0.07, q_vir<1.35 )'])
#ind_rel_Neto.to_csv(path_out+'halo_properties/relaxed_halos_Neto.dat',index=False)

if __name__ == '__main__':
    print('test')