#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:14:45 2021

@author: guillaume
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator,LogLocator)
import pandas as pd
import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()



def plot_beauty_simple(pos3d,mass,r_max=1,
                       r_marge=0.2,R_box=6,ax_1=0,ax_2=1,name_add=''):
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
    a,b = np.mean(fullbox)*0.05, np.max(fullbox)/2 # Arturo's guess
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
    #dr = r_marge/2
    #r_sphere = R_box + dr
    #ax.set_title(r'halo selection $|\phi| >  |\overline{\phi(}r='+str()+'{:.2f}'.format(r_sphere)+'\pm {:.2f}'.format(dr)+' R_{vir})|$')
    #ax.set_title('$s = 0.04$, $q = 1.04$, $S = 0.69$, $Q = 0.9$, $T = 0.38$')
    #ax4.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    #plt.savefig(''+path+'plot/'+name+'threshold_radius_shell_{:.1f}'.format(R_box)+'_Rvir'+name_add+'.pdf')
    plt.show()
    #plt.close(fig)  
    return()


def scatter_plot(pos_use,pot_use) :
    plt.rc('font', family='serif')
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    fig, ax = plt.subplots(figsize=[10,10]) # plot
    ax.tick_params(which='both', width=0.5, direction="in",
                   labeltop=True,labelbottom=False, labelleft=False, labelright=True)
    ax.scatter(pos_use[:,0],pos_use[:,2],c=pot_use,s=0.01)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    plt.show()
    return()

if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/test_halo_from_data/' # save particle positions of the created halo
    path_data = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/data/'
    #path_prop = '../../../../DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/properties/relaxed_halos/'
    path_prop = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_14_5_sphere_sel/properties/'
    path_test = '../../../../DEUSS_648Mpc_2048_particles/output_not_MF/test_halo_from_data/'
    #path_MF = '../../../../DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/input_MF_1000_part/relaxed_halos/halo_created_from_ellipsoidal_profile_of_data_variation/'
    path_MF = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_14_5_sphere_sel/data_MF/ellipsoid_from_data_no_variation_with_sphere_check/'
    names_array = np.loadtxt(path_prop+'names_all.dat',dtype=str)
    prop_array = np.loadtxt(path_prop+'prop_all.dat')
    indices = np.loadtxt(path_prop+'indices_non_shifted_haloes.dat')
    indices = np.array(indices,dtype=int)
    prop_array = prop_array[indices]
    names_array = names_array[indices]
    #abc_array = np.loadtxt(path_prop+'abc_non_shifted_haloes.dat')
    #Passage_array = np.loadtxt(path_prop+'Passage_rotation_non_shifted_haloes.dat')
    abc_array = np.array((pd.read_csv(path_prop+'abc_non_shifted_haloes.dat')))[:,1:]
    Passage_array = np.array((pd.read_csv(path_prop+'Passage_rotation_non_shifted_haloes.dat')))[:,1:]
    print('abc =',abc_array[0])
    #print(Passage_array[0])
    Rvir_array = prop_array[:,0]
    #Rvir_array = prop_array[:,0]
    print(prop_array[0])
    cdm_fof = prop_array[:,2:5]
    cdm_dens = prop_array[:,5:9]
    shift_array = np.sqrt( (cdm_fof[:,0] - cdm_dens[:,0])**2 + (cdm_fof[:,1] - cdm_dens[:,1])**2 + (cdm_fof[:,2] - cdm_dens[:,2])**2 ) 
    shift_array /= Rvir_array
    h_sml = 5*10**(-3)
    res = 3 * h_sml # in Mpc/h
    r_max, m_part = 1, 1
    N_bin_profile = 30 # why 20 ?
    N_times = 1
    #N_halos = len(Rvir_array)
    N_halos = 1
    N_part_MF = 1000
    my_push = 0.4
    my_ax = 2
    real_data = True
    N_bin = 6
    error_halo = np.zeros((N_halos,4))
    for k in range(1,2) :
        print('k =',k)
        print('shift =',shift_array[k])
        print(names_array[k])
        if real_data == True : # constant ellipsoid
            a_1, a_2, a_3 = abc_array[k,0], abc_array[k,1], abc_array[k,2]
            print('abc =',a_1, a_2, a_3)
            Passage = Passage_array[k]
            Passage = Passage.reshape((3,3))
            name_file = names_array[k]
            name_use = name_file.replace('.dat','_sphere_sel_1_Rvir.dat')
            Rvir = Rvir_array[k]
            pos = np.loadtxt(path_data+'sphere_sel_1_Rvir/'+name_use) # in Mpc/h
            r_min = res/Rvir # in Rvir unit
        pos /= Rvir # in Rvir unit
        x, y, z = pos[:,0], pos[:,1], pos[:,2] # positions in Rvir unit in the simulation box frame
        #print(len(x))
        radius = np.sqrt(x**2 + y**2 + z**2)
        ind_r = np.where(radius < 1)
        x, y, z = x[ind_r], y[ind_r], z[ind_r]
        #print(len(x))
        N_part_tot = len(x)
        pos = np.matrix(np.array((x,y,z)))
        pos = np.array((np.dot(np.linalg.inv(Passage),pos)))  
        #print(pos)
        pos3d = pos.transpose() 
        pos = np.array(np.reshape(pos3d,3*N_part_tot),dtype=np.float32) 
        #print(pos)
        mass_array = np.ones((N_part_tot),dtype=np.float32)/np.float32(N_part_tot)
        
        ok, acc, potential = cf.getGravity(pos, mass_array, h_sml/Rvir, 
                                           G=1.0, theta=0.6, kernel_type=1, ncrit=6)
        #plot_beauty_simple(pos3d,np.abs(potential))
        #print(potential)
        ind_sphere = np.where(radius < 0.5)[0]
        print(len(ind_sphere))
        pos3d = pos3d[ind_sphere]
        potential = potential[ind_sphere]
        M_vir = N_part_tot
        print(N_part_tot)
        pot_vir = 1 * M_vir/1
        Rvir_c = a_3
        Nvir_c = len(np.where(radius < a_3)[0])
        fac_c = Nvir_c/(N_part_tot*Rvir_c)
        print(fac_c)
        i = 1
        for fac in [1,2,3] :
            pot_max = fac 
            ind_pot = np.where(np.abs(potential) > pot_max)[0]
            print(len(ind_pot))
            pot_use = np.abs(potential)[ind_pot]
            pos_use = pos3d[ind_pot]
            scatter_plot(pos_use,pot_use)
            #plot_beauty_simple(pos_use,pot_use)
            i += 1
        #print(pot_vir)
        plt.figure(0)
        plt.scatter(radius[ind_sphere],np.abs(potential),s=0.01)
        plt.show()
        
        #plt.figure(2)
        #plt.scatter(x,z,s=0.01)
        #plt.show()
        
        cdm_new = np.array((np.mean(x),np.mean(y), np.mean(z)))
        potential_max = np.max(np.abs(potential))
        ind_dens = np.where(np.abs(potential) == potential_max)
        cdm_new_dens = np.array((x[ind_dens],y[ind_dens],z[ind_dens]))
        #print(cdm_new)
        #print(cdm_new_dens)
        shift = np.sqrt(cdm_new[0]**2 + cdm_new[1]**2 + cdm_new[2]**2 )
        print('shift =',shift)