#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:10:23 2020

@author: guillaume
"""

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from matplotlib.colors import LogNorm
from random import sample
from scipy.spatial.distance import pdist
import ctypes as ct
from scipy.spatial import ConvexHull
import random
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator,LogLocator)
import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()

from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import L_box, hsml, G_Newton, v_ren



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


# very fast code to find if a point is in the hull, with a tolerance
# found here: 
# https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl/16898636
# In words, a point is in the hull if and only if for every equation (describing the facets) 
# the dot product between the point and the normal vector (eq[:-1]) plus the offset (eq[-1]) is less than or equal to zero.
# You may want to compare to a small, positive constant tolerance = 1e-12 rather than to zero because of issues of numerical precision 
# (otherwise, you may find that a vertex of the convex hull is not in the convex hull).
# The convex hull of a point set P is the smallest convex set that contains P.
# If P is finite, the convex hull defines a matrix A and a vector b such that for all x in P, Ax+b <= [0,...]" 
# The rows of A are the unit normals; the elements of b are the offsets. 
def point_in_hull(hull, point, tolerance):
    #print('point in hull')
    # hull is the convex hull of my points set
    # equations gives me the normal vectors to the hull and the offsets, for each point of the hull
    #print(hull.equations)
    #print(hull.equations[0])
    #print(len(hull.equations))
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
               for eq in hull.equations)

# found here: 
# https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl/16898636
# pos and R_box should be in Rvir unit
def random_in_hull(pos,my_push,array_min,R_box,N_part,time_limit):
    hull = ConvexHull(pos) # in Rvir unit
    t0 = time.time()
    p = 1
    my_random_points = np.zeros((N_part,3))
    time_p = time.time() - t0
    while p <= N_part :
        if time_p > time_limit :
            break
        else :
            x_p = random.uniform(-R_box+my_push-array_min[0],R_box+my_push-array_min[0])
            y_p = random.uniform(-R_box+my_push-array_min[1],R_box+my_push-array_min[1])
            z_p = random.uniform(-R_box+my_push-array_min[2],R_box+my_push-array_min[2])
            my_point = [x_p,y_p,z_p] # in Rvir unit
            tolerance = 1e-12
            in_hull_bool = point_in_hull(hull,my_point,tolerance)
            if in_hull_bool == True :
                my_random_points[p-1] = my_point # in Rvir unit
                p += 1
                t0 = time.time()
                time_p = time.time() - t0
            else :
                time_p = time.time() - t0
    return(my_random_points,hull)
    
def random_in_hull_sphere(my_push,array_min,R_box,N_part,time_limit):
    t0 = time.time()
    p = 1
    my_random_points = np.zeros((N_part,3))
    time_p = time.time() - t0
    while p <= N_part :
        if time_p > time_limit :
            break
        else :
            x_p = random.uniform(-R_box,R_box) # in Rvir unit
            y_p = random.uniform(-R_box,R_box)
            z_p = random.uniform(-R_box,R_box)
            r_p = np.sqrt(x_p**2 + y_p**2 + z_p**2)
            if r_p <= R_box :
                my_point = [x_p,y_p,z_p] # in Rvir unit
                my_random_points[p-1] = my_point # in Rvir unit
                p += 1
            time_p = time.time() - t0
    my_random_points = my_random_points + my_push - array_min
    return(my_random_points)
      
def random_in_hull_cube(pos,N_part,R_box,time_limit):
    t0 = time.time()
    p = 1
    my_random_points = np.zeros((N_part,3))
    time_p = time.time() - t0
    while p <= N_part :
        if time_p > time_limit :
            break
        else :
            x_p, y_p, z_p = random.uniform(-R_box,R_box), random.uniform(-R_box,R_box), random.uniform(-R_box,R_box)
            my_point = [x_p,y_p,z_p] # in Rvir unit
            my_random_points[p-1] = my_point # in Rvir unit
            p += 1
            time_p = time.time() - t0
    return(my_random_points)

def compute_dis_in_c(N_part, pos):
    '''This function computes the distance between particle position contained in pos
    It is using the function compute_dis in the C code distance.c, calling the extension .so
    The particle number is N_part. posshould then be of size 3*N_part such as
    pos = np.array((x1,y1,z1,x2,y2,z2,...),dtype=np.float64)
    It returns the min, the max and the mean distance.
    '''
    # to compile my c function
    # gcc -fPIC -shared -o libdist.so distances.c -lm
    # absolute path to the C code in .so
    path_abs = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/Data_analysis/distances_in_C/'
    #name_so = 'libdist.so'
    name_so = 'libdist_hop_ne.so' # my computer # 'libdisnews.so' # the cluster
    # I Import the C code containing the function I want to use, the .so is for Python
    distances = ct.CDLL(''+path_abs+''+name_so+'')
    # C-type corresponding to numpy array of np.float64, pointer
    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
    # define prototypes: for input of the C function compute_dis
    distances.compute_dis.argtypes = [ct.c_size_t, ND_POINTER_1]
    # As the output of my function is a structure (in C, it returns 3 doubles)
    # I build a class to allow the interpretation of the output of the C function compute_dis
    # Python representation of the C struct re_val
    class ReVal(ct.Structure):
        _fields_ = [("dis_min", ct.c_double),("dis_max", ct.c_double),("dis_mean", ct.c_double)]
        
    distances.compute_dis.restype = ReVal # output of the C function compute_dis
    my_dis = distances.compute_dis(N_part, pos) # use the C function
    # use the class ReVal to get the output
    dis_min, dis_max, dis_mean = my_dis.dis_min, my_dis.dis_max, my_dis.dis_mean 
    return(dis_min, dis_max, dis_mean)


def plot_beauty_simple(path,name,pos3d,mass,r_max,
                       r_marge=0.2,R_box=6,ax_1=0,ax_2=2,name_add=''):
    ''' Arturo's function for a kind of 2D scatter plot with a weigh, 
    such as, gravitational potential.
    '''
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.right'] = False
    plt.rc('font', family='serif')
    # here you define a figure fig and inside a frame ax, this can be scale for more frames
    fig, ax = plt.subplots(figsize=[10,10])
    plt.rcParams['font.size'] = 20
    # this is a bit innesesary but i do it when i want to rotate but it is an array with a vector [x,y,z] for each particles
    nupos = pos3d
    #print(len(nupos))
    # the number of bins in x and y 
    pixels = 1000
    # limits for the data you want to consider for the histogram
    #print('plot')
    #thelimmin,thelimmax = pos3d.min(),pos3d.max()
    # this defines the limit of the frame
    #R_box = np.int(2*Rmax)
    ax.set_xlim([-r_max,r_max])
    ax.set_ylim([-r_max,r_max])
    #this will be a squared plot so I define the same array for x and y
    edges = np.linspace(-r_max,r_max,pixels)
    # this is a way of make it easier if you want to see a different projection than the plan x-y
    #x,y,z=0,1,2
    # this is the histogram one by numpy 
    H, xedges, yedges = np.histogram2d(nupos[:,ax_1],# the data in the x axis 
                                   nupos[:,ax_2],# the data for the y axis
                                   bins=(edges, edges), # the bins for boths sets of data 
                                   weights=mass# here you wright your data it can be any array as long as it has the same length as x and y
                                  )
    # this is stupid but is numpy's fault you need to take the transposed of the matrix that represent the 2dhistogram
    fullbox = H.T
    # this is arbitrary you will see
    a,b = np.mean(fullbox)*0.001, np.max(fullbox)/4 # Arturo's guess
    #a,b = np.mean(fullbox)*0.00001, np.max(fullbox)/2
    #a,b = np.min(mass), np.max(mass)
    #a,b = np.min(fullbox), np.max(fullbox)
    # now we plot
    mass_2 = ax.imshow(fullbox+0.1,# the added number is to have all bins with color
                   interpolation='gaussian',# this defines how you plot frontiers between adjacent bins
                   origin='lower', # if you dont do this you get a weird orientation on the plot
                   cmap="magma", # here is your choice for the color map
                   extent=[edges[0], edges[-1], edges[0], edges[-1]], # the limits
                   norm=LogNorm(vmin=a,vmax=b), # this is a trick, you have to import "from matplotlib.colors import LogNorm" so here you use the values I defined as limits but is by playin that i know this|
                  )
    ax.set_xlabel('x ($R_{vir}$)')
    ax.set_ylabel('y ($R_{vir}$)',labelpad=0)
    ax.tick_params(which='major', length=14) # I set the length
    ax.tick_params(which='minor', length=7)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    #f'{number:9.4f}
    #{:.2f}'.format(enrichment)
    dr = r_marge/2
    r_sphere = R_box + dr
    #ax.set_title(r'halo selection $|\phi| >  |\overline{\phi(}r='+str()+'{:.2f}'.format(r_sphere)+'\pm {:.2f}'.format(dr)+' R_{vir})|$')
    #ax.set_title('$s = 0.04$, $q = 1.04$, $S = 0.69$, $Q = 0.9$, $T = 0.38$')
    #ax4.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    plt.savefig(''+path+'plot/'+name+'threshold_radius_shell_{:.1f}'.format(R_box)+'_Rvir'+name_add+'.pdf')
    plt.show()
    #plt.close(fig)    
    
def compute_q(phi,vel):
    # set pos in Mpc/h and vel in km/s
    vx, vy, vz = vel[:,0], vel[:,1], vel[:,2]
    N_part = len(vx)
    K_all = (vx**2 + vy**2 + vz**2)/2
    K_tot = np.sum(K_all)
    #mass_array = np.ones((N_part),dtype=np.float32)
    #ok, acc, phi = cf.getGravity(pos, mass_array, hsml, 
    #                                   G=1.0, theta=0.6, kernel_type=1, ncrit=6)
    # I reset the potential gravitational energy in physical unit
    V_all = phi * G_Newton
    # I compute the potential energy of my halo in (Megameter/S)**2
    V_tot = np.sum(V_all) # facteur 1/2 formule
    q = 2*K_tot/np.abs(V_tot)
    return(q)
    
    
def prepare_data_for_MF(pos,my_push):
    '''This function prepare the data position for MF.'''
    array_min = np.array((np.min(pos[:,0]), np.min(pos[:,1]), np.min(pos[:,2])))
    pos = pos - array_min 
    pos = pos + np.array((my_push,my_push,my_push))
    return(pos,array_min)


def perform_iso_for_MF(path, name, data, cdm, Rvir, N_r, multiple_phi=False, r_min=0.1, step=0.1,
                       sphere=False, R_sphere=1, N_part_MF=1000, N_sel_MF=1,time_limit=1000):
    '''This function computes isopotential for MF.
    '''
    x, y, z = data[0], data[1], data[2] # box unit
    x, y, z, bad_periodicity = periodicity(x,y,z,cdm/L_box)
    x, y, z = x*L_box, y*L_box, z*L_box # Mpc/h
    N_part_tot = len(x)
    x, y, z = x-cdm[0], y-cdm[1], z-cdm[2] # Mpc/h, centered
    radius = np.sqrt( x**2 + y**2 + z**2 ) # Mpc/h
    ind_r = np.argsort(radius)
    r_sorted = radius[ind_r] # Mpc/h
    x, y, z = x[ind_r], y[ind_r], z[ind_r] # Mpc/h, centered
    data = data[:,ind_r]
    pos = np.array((x,y,z)) # Mpc/h
    pos3d = pos.transpose() # Mpc/h
    pos = np.array(np.reshape(pos3d,3*N_part_tot),dtype=np.float32) # Mpc/h
    mass_array = np.ones((N_part_tot),dtype=np.float32)
    ok, acc, potential = cf.getGravity(pos, mass_array, hsml, 
                                       G=1.0, theta=0.6, kernel_type=1, ncrit=6)
    pot = np.abs(potential)  
    name_add = ''
    if sphere == True : # if True, I add a spherical selection on the particles
        ind_sphere = np.where(r_sorted < R_sphere*Rvir)[0] # R_sphere is in Rvir unit, and I setit in Mpc/h
        r_sorted = r_sorted[ind_sphere] # Mpc/h
        pot = pot[ind_sphere]
        pos3d = pos3d[ind_sphere] # Mpc/h
        data = data[:,ind_sphere]
        name_add = '_sphere_'+str(R_sphere)+''
    N_r_beg = N_r - 1
    if multiple_phi == True : # if True, I will compute the selections for many phi_threshold
        N_r_beg = 0
    N_part_in_sel = np.zeros((N_r-N_r_beg),dtype=int)
    pot_threshold, my_diameter = np.zeros((N_r-N_r_beg)), np.zeros((N_r-N_r_beg))
    for i in range(N_r_beg,N_r):
        print(i)
        threshold = r_min + i*step # in Rvir unit
        print(threshold)
        ind_moins = np.where(r_sorted < (threshold + step)*Rvir)[0]
        ind_shell = np.where(r_sorted[ind_moins] > threshold*Rvir)[0]
        pot_shell = pot[ind_moins][ind_shell]
        pot_threshold[i-N_r_beg] = np.mean(pot_shell)
        print(pot_threshold[i-N_r_beg])
        ind_pot = np.where(pot > pot_threshold[i-N_r_beg])[0]
        pot_sel = pot[ind_pot]
        r_max = r_sorted[ind_pot][-1]/Rvir # in Rvir unit
        print(r_max)
        #np.savetxt(''+path+'phi_full/'+name+'_phi_sel_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'.dat',pot_sel)
        pos3d_sel = pos3d[ind_pot] # in Mpc/h
        print(pos3d_sel)
        data_sel = data[:,ind_pot]
        #np.savetxt(''+path+'pos_full/'+name+'_pos_sel_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'.dat',data_sel)
        N_part_sel = len(ind_pot)
        N_part_in_sel[i-N_r_beg] = N_part_sel
        print(N_part_sel)
        my_diameter[i-N_r_beg] = 2 * (threshold + step) * Rvir # old version, in Mpc/h
        #distances_sel = pdist(pos3d_sel)
        #my_diameter[i] = np.max(distances_sel) # new version, diameter of the selection in Mpc/h
        #t_dis_beg = time.time()
        #my_pos_for_c = np.reshape(pos3d_sel,3*N_part_sel)
        #dis_min, dis_max, dis_mean = compute_dis_in_c(N_part_sel, my_pos_for_c)
        #t_dis_end = time.time()
        #print('time computation of c distance in second')
        #print(t_dis_end - t_dis_beg)
        #my_diameter[i-N_r_beg] = dis_max
        #print('c distance')
        #print(my_diameter[i-N_r_beg])
        my_push = my_diameter[i-N_r_beg]/5 # in Mpc/h
        pos_MF_full, array_min = prepare_data_for_MF(pos3d_sel,my_push) # all in Mpc/h
        N_part = len(pos_MF_full)
        print('N_sel')
        for j in range(N_sel_MF):
            print(j)
            my_integers = sample(range(N_part),N_part_MF)
            pos_MF_save = pos_MF_full[my_integers] # Mpc/h
            t_h =time.time()
            my_random_points, hull = random_in_hull(pos_MF_full,my_push,array_min,r_max*Rvir,N_part_MF,time_limit)
            my_random_points_sphere = random_in_hull_sphere(my_push,array_min,threshold*Rvir,N_part_MF,time_limit)
            t_h_end = time.time()
            print(t_h_end-t_h)
            if N_sel_MF == 1 :
                print('ok')
                np.savetxt(''+path+'data_MF/'+name+'_pos_MF_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'.dat',pos_MF_save)
                #np.savetxt(''+path+'data_MF/'+name+'_random_hull_MF_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'.dat',my_random_points)
                #np.savetxt(''+path+'data_MF/random_sphere_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'.dat',my_random_points_sphere)
            else :
                print('hop')
                np.savetxt(''+path+'data_MF/'+name+'_pos_MF_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'_'+str(j)+'.dat',pos_MF_save)
                #np.savetxt(''+path+'data_MF/'+name+'_random_hull_MF_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'_'+str(j)+'.dat',my_random_points)
                #np.savetxt(''+path+'data_MF/random_sphere_radius_{:.1f}'.format(threshold)+'Rvir'+name_add+'_'+str(j)+'.dat',my_random_points_sphere)
        #np.savetxt(''+path+'data_isopotential/pos_sel_radius_{:.1f}'.format(threshold)+'Rvir.dat',pos3d_sel)
        #np.savetxt(''+path+'data_isopotential/pos_MF_radius_{:.1f}'.format(threshold)+'Rvir.dat',pos_MF)
        #np.savetxt(''+path+'data_isopotential/phi_sel_MF_radius_{:.1f}'.format(threshold)+'Rvir.dat',mass_sel)
        #plot_beauty_simple(path,name,pos3d_sel/Rvir,pot_sel,r_max,r_marge=step,R_box=threshold,name_add=name_add)
    print(N_part_in_sel)
    #np.savetxt(''+path+'phi_threshold_and_N_part/N_part_in_sel_'+name+''+name_add+'.dat',N_part_in_sel)
    #np.savetxt(''+path+'phi_threshold_and_N_part/potential_threshold_'+name+''+name_add+'.dat',pot_threshold)
    #np.savetxt(''+path+'phi_threshold_and_N_part/diameters_'+name+''+name_add+'.dat',my_diameter) 
    return(pos3d_sel, pot_sel, N_part_in_sel, pot_threshold,pos_MF_save,my_random_points,my_random_points_sphere)

t_in_in = time.time()
#n = np.int(sys.argv[1])
#n=1
#halo = np.int(sys.argv[1])
halo = 1
def indice_cluster(n_parallel):
    # if I want to apply to the cluster
    #N_haloes = 2876 # Number of haloes
    #n_use = N_haloes // N_cut
    N_cut = 107
    N_haloes = 328
    multiple = 5
    if n_parallel < N_cut :
        f_begin = (n_parallel - 1) * multiple #n_use #np.int(multiple*(n-1)) # begining and end of my loop
        f_end = n_parallel * multiple #n_use #np.int(multiple*n)
    else :
        f_begin = (n_parallel - 1) * multiple #n_use #multiple*N_cores
        f_end = N_haloes
    return(f_begin,f_end)
f_begin,f_end = indice_cluster(halo)
f_begin, f_end = 0, 10
print(f_begin,f_end)
#halo = 2
#halo = 36
#path = '../mass_bin_data/mass_10_14_Msun_FOF/halo_properties/'
#names = np.genfromtxt(''+path+'haloes_usefull_'+str(n)+'.dat',dtype='str')
mass_log = 14.5
path_names = '../mass_bin_data/mass_10_'+str(mass_log)+'_Msun_beyond_FOF/halo_properties/' # be carfull, I need to look for the data on the cluster
names = np.genfromtxt(''+path_names+'names_sel_random.dat',dtype='str')
print(names)
path_prop = '../mass_bin_data/mass_10_'+str(mass_log)+'_Msun_beyond_FOF/halo_properties/'
prop = np.loadtxt(''+path_prop+'prop_sel_random.dat')            
print(prop) 
#N_haloes = len(names)
#N_haloes = 1
path = '../mass_bin_data/mass_10_'+str(mass_log)+'_Msun_beyond_FOF/'
path_out = '../input_MF/data_halo/isopotential_selection/mass_10_'+str(mass_log)+'_Mvir/'
#step = 0.1
N_r = 10
#threshold = N_r*step
for i in range(f_begin,f_end):
    name = names[i]
    #name = str(names)
    data = np.loadtxt(''+path+'data/'+name+'_5_Rvir_new.dat')
    #prop = np.loadtxt(''+path+'halo_properties/data_properties_'+str(n)+'.dat')  
    #prop = np.loadtxt(''+path+'halo_properties/data_properties_full.dat')             
    cdm = np.array((prop[i,5],prop[i,6],prop[i,7]))
    #cdm = np.array((prop[i,5],prop[6],prop[7]))
    Rvir = prop[i,0]
    #Rvir = prop[0]
    print(name)
    print(cdm)
    print(Rvir)
    pos_last_sel, pot_last_sel, N_part_in_sel, pot_threshold,pos_MF_save,my_random_points, my_random_points_sphere = perform_iso_for_MF(path_out,name,data,cdm,Rvir,
                                                                                                                                        N_r,multiple_phi=False,sphere=True,R_sphere=1.2)
    #np.savetxt(''+path_out+'phi_threshold_and_N_part/N_part_in_sel_'+name+'_sphere.dat',N_part_in_sel)
    #np.savetxt(''+path_out+'phi_threshold_and_N_part/potential_threshold_'+name+'_sphere.dat',pot_threshold)
    #np.savetxt(''+path+'data_isopotential/pos_sel_radius_{:.1f}'.format(threshold)+'Rvir.dat',pos3d_sel)
    #path = '../input_MF/data_halo/isopotential_selection/'
    #np.savetxt(''+path+'data_MF/pos_MF_'+name+'_all_radii.dat',pos_MF)
    #np.savetxt(''+path+'data_isopotential/phi_sel_MF_radius_{:.1f}'.format(threshold)+'Rvir.dat',mass_sel)
    #plot_beauty_simple(path_out,name,pos_last_sel/Rvir,pot_last_sel,r_marge=step,R_box=threshold )
    
t_end_end = time.time()
print(t_end_end - t_in_in)
#data = np.loadtxt(''+path+'data_isopotential/pos_MF_radius_{:.1f}'.format(threshold)+'Rvir.dat')
#data = np.loadtxt(''+path+'data_isopotential/pos_MF_radius_1.5Rvir.dat')
#print(data)
#plt.figure(6)
#plt.scatter(pos_last_sel[:,0],pos_last_sel[:,1],c='r',s=0.01)
#plt.scatter(my_random_points[:,0],my_random_points[:,2],c='b')
#plt.scatter(my_random_points_sphere[:,0],my_random_points_sphere[:,2],c='g')
#plt.show()