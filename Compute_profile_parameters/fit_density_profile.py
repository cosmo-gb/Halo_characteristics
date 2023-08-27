#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:04:38 2021

@author: guillaume
"""
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator,LogLocator)
import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()
from random import sample
import pandas as pd

from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import L_box

def rotate_position(pos,Passage) :
    # This function rotates the particle positions,
    #  the natural frame of the ellipsoid (x <=> a_1, y <=> a_2, z <=> a_3)
    N_part = len(pos)
    print('N_part_tot initial =',N_part)
    x, y, z = pos[:,0], pos[:,1], pos[:,2] # positions in Rvir unit in the simulation box frame
    position = np.matrix(np.array((x,y,z)))
    position_new = np.array((np.dot(np.linalg.inv(Passage),position)))
    position_new = position_new.transpose() #reshape((N_part,3))
    # positions in the axis frame
    x_new, y_new, z_new = position_new[:,0], position_new[:,1], position_new[:,2]
    return(x_new,y_new,z_new)


def do_plot_profile(r_bin,n_bin,r_bin_2,n_bin_2,r_loc,n_loc,
                    r_and_rho_loc_in,r_and_rho_loc_out,path_and_name,res_sphere) :
    r_loc_mean_in, n_loc_mean_in = r_and_rho_loc_in[0:2]
    n_loc_std_in, n_loc_std_minus_in, n_loc_std_plus_in = r_and_rho_loc_in[2:]
    r_loc_mean_out, n_loc_mean_out = r_and_rho_loc_out[0:2]
    n_loc_std_out, n_loc_std_minus_out, n_loc_std_plus_out = r_and_rho_loc_out[2:]
    fac = 2 # to see better for the plot
    plt.rc('font', family='serif')
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    fig, ax = plt.subplots() # plot
    ax.tick_params(which='both', width=0.5, direction="in",
                   labeltop=True,labelbottom=False, labelleft=False, labelright=True)
    ax.vlines(res_sphere,np.min(n_bin)/fac,np.max(n_bin)*fac,ls='--',color='k',label='resolution (sphere)')
    ##################################################################################################################
    ax.loglog(r_loc_mean_in,n_loc_mean_in,c='orange',ls='-.',lw=3,label='data loc mean')
    ax.fill_between(r_loc_mean_in,n_loc_mean_in-n_loc_std_minus_in,n_loc_mean_in+n_loc_std_plus_in,color='orange',alpha=0.2)
    ax.loglog(r_loc[0],rho_loc[0],c='orange',ls='',marker='.',markersize=1,label='data loc')
    ax.loglog(r_loc_mean_out,n_loc_mean_out,c='lightgreen',ls='-.',lw=1,label='create loc mean')
    ax.fill_between(r_loc_mean_out,n_loc_mean_out-n_loc_std_minus_out,n_loc_mean_out+n_loc_std_plus_out,color='lightgreen',alpha=0.2)
    ax.loglog(r_loc[1],rho_loc[1],c='lightgreen',ls='',marker='.',markersize=1,label='create loc')
    ##################################################################################################################"
    
    ax.loglog(r_bin[0],n_bin[0],c='r',ls='--', lw=3, marker='o',markersize=6,label='data sphere')   
    ax.loglog(r_bin[1],n_bin[1],c='c',ls='-', lw=3, marker='o',markersize=6,label='data ellipsoid')  
    ind_not_zeros = np.where(r_bin_2[0] != 0)[0]
    ax.loglog(r_bin_2[0][ind_not_zeros],n_bin_2[0][ind_not_zeros],c='g',ls='--', lw=1, marker='o',markersize=3,label='creation sphere')   
    ax.loglog(r_bin_2[1],n_bin_2[1],c='b',ls='-', lw=1, marker='o',markersize=3,label='creation ellipsoid')
    #ax.loglog(r_bin_2[2],n_bin_2[2],c='gray',ls='--', marker='',markersize=3,label='sphere old renormalized')    
    #ax.fill_between(r_bin,n_bin-drho,n_bin+drho,color='orange')
    ax.set_ylim(np.min(n_bin[1])/fac,np.max(n_bin[0])*fac)
    ax.set_xlim(np.min(r_bin[1])/fac,np.max(r_bin[0])*fac)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    ax.set_xlabel('$r$ or $r_{el}$ $[R_{vir}]$',labelpad=2)
    ax.set_ylabel('$n(r)$ $[R_{vir}^{-3}]$',labelpad=0)
    ax.legend(loc='lower left', prop={'size': 10}, ncol=2, frameon=False)   
    #concentration = R[0]/r_minus_2[0]
    '''ax.set_title(r'$\rho(r) = \rho_s$ $x^{-\gamma} (1 + x^{\alpha})^{(\gamma - \beta)/\alpha}$, $x=r/r_s$,'+ # profile
    # parameters
    r' $\alpha = '+str(np.round(alpha,2))+r'$, $\beta = '+str(np.round(beta,2))+r'$, $\gamma = '+str(np.round(gamma,2))+
    '$, $N_{halo} = '+str(np.round(N_part,1))+  r'$, $c = '+str(np.round(concentration,2))+'$',
    fontsize=10)'''
    #plt.savefig(path_and_name,format='pdf',transparent=True)
    plt.show()
    #plt.close(fig)
    return()

def do_plot_profile(r_bin,rho_bin,rho_abg,rho_NFW,name):
    #plt.rcParams.update({'font.size': 20})
    plt.rc('font', family='serif',size=20)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    fig, ax = plt.subplots() # plot
    ax.tick_params(which='both', width=0.5, direction="in",
                   labeltop=True,labelbottom=False, labelleft=False, labelright=True)
    ax.loglog(r_bin,rho_bin,c='b',ls='-',marker='.',markersize=10,lw=1,label='N-body')
    ax.loglog(r_bin,rho_abg,c='orange',ls='-',marker='.',markersize=5,lw=1,label=r'fit $(\alpha,\beta,\gamma)$')
    ax.loglog(r_bin,rho_NFW,c='g',ls='-',marker='.',markersize=3,lw=1,label='fit NFW')
    ax.set_xlabel('$r$ $[R_{vir}]$',labelpad=2)
    ax.set_ylabel('$n(r)$ $[R_{vir}^{-3}]$',labelpad=0)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    ax.legend(loc='lower left', ncol=1, frameon=False)  
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/test_N_part_and_example/plots/'
    plt.savefig(path+name+'_n_sphere.pdf',format='pdf',transparent=True)
    plt.show()
    return()

def do_plot_profile_r_2(r_bin,rho_bin,rho_abg,rho_NFW,name):
    plt.rc('font', family='serif')
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    fig, ax = plt.subplots() # plot
    ax.tick_params(which='both', width=0.5, direction="in",
                   labeltop=True,labelbottom=False, labelleft=False, labelright=True)
    ax.loglog(r_bin,rho_bin*r_bin**2,c='b',ls='-',marker='.',markersize=10,lw=1,label='N-body')
    ax.loglog(r_bin,rho_abg*r_bin**2,c='orange',ls='-',marker='.',markersize=5,lw=1,label=r'fit $(\alpha,\beta,\gamma)$')
    ax.loglog(r_bin,rho_NFW*r_bin**2,c='g',ls='-',marker='.',markersize=3,lw=1,label='fit NFW')
    ax.set_xlabel('$r$ $[R_{vir}]$',labelpad=2)
    ax.set_ylabel('$n(r)$ $r^2$ $[R_{vir}^{-1}]$',labelpad=0)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    ax.legend(loc='lower right', ncol=1, frameon=False)   
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/test_N_part_and_example/plots/'
    plt.savefig(path+name+'_n_r_2_sphere.pdf',format='pdf',transparent=True)
    plt.show()
    return()

def profile(r_sorted,N_part_bin,N_part_tot,factor_random=1,r_min=0):
    # r_sorted should be in Mpc/h
    # N_part_bin is the number of particles in each shells (fixed)
    # N_part_tot is the total number of particles in the halo
    # factor_random: take into account the mass differnce between a particle mass in the halo 
    # and a particle in the simulation DEUS, it can be =1 if I am using data from DEUS
    N_bin = np.int(N_part_tot/N_part_bin)
    r_bin_mean = np.zeros((N_bin))
    r_bin_sqrt = np.zeros((N_bin))
    v_shell = np.zeros((N_bin))
    r_shell = np.zeros((N_bin + 1))
    r_shell[0] = r_min
    d_N_part_bin = np.sqrt(N_part_bin)
    for b in range(N_bin):
        r_shell[b+1] = r_sorted[(b+1)*N_part_bin-1]
        v_shell[b] = (4*np.pi/3) * (r_shell[b+1]**3 - r_shell[b]**3)
        r_bin_mean[b] = np.mean(r_sorted[b*N_part_bin:(b+1)*N_part_bin])
        r_bin_sqrt[b] = np.std(r_sorted[b*N_part_bin:(b+1)*N_part_bin])
    rho_bin = N_part_bin/v_shell
    d_rho_bin = d_N_part_bin/v_shell
    rho_bin = rho_bin * factor_random
    # r_bin_mean: averaged of the particles radii in each shells in Mpc/h
    # rho_bin: profile in each shells, mass of each shell divided by its volume, in rho_crit unit
    # r_bin_sqrt : sqrt of radius in each shells in Mpc/h
    return(r_bin_mean,rho_bin,r_bin_sqrt)

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
    ax.set_ylabel('z ($R_{vir}$)',labelpad=0)
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
    #plt.savefig(path+name+'.pdf')
    plt.show()
    #plt.close(fig)    
    return()

def profile_log_r_bin_hist(radius,
                           r_min=0.01,r_max=1,N_bin=30,
                           factor_random_mass=1):
    # It computes the density profile with an equal logarithmic shell size
    # It imposes shells of a given size, computed with r_min r_max and N_bin
    # Then it computes the number of particles in each shell
    # the density is simply N_part/V_shell
    # radius is the particles radii, not need to be sorted
    # radius, r_min and r_max should be in the same length unit
    # N_bin is the number of shells used for the density profile
    # rho_loc should be sorted as radius and in rho_crit unit (optional, to compute the scatter of the profile)
    # factor_random: take into account the mass differnce between a particle mass in the halo 
    # and a particle in the simulation DEUS, it can be =1 if I am using data from DEUS
    # r_bin_log: contains N_bin elements, radius_mean of each shell
    # rho_bin_log: contains N_bin elements, density profile in the shells
    # N_part_in_shell: contains N_bin elements, number of particles in each shell
    if r_min <= 0 :
        print('r_min should be strictly positive')
    r_log_bin = np.logspace(np.log10(r_min),np.log10(r_max),N_bin+1) 
    N_part_in_shell, r_shell = np.histogram(radius, bins=r_log_bin) 
    r_log_bin = (r_log_bin[1:] + r_log_bin[:-1])/2
    Volume_shell = (4*np.pi/3)*(r_shell[1:]**3 -r_shell[:-1]**3)
    rho_log_bin = N_part_in_shell*factor_random_mass/Volume_shell
    return(r_log_bin,rho_log_bin,N_part_in_shell) 

def do_scatter(pos_fof,N_fof,name,ax_1=0,ax_2=1,N_choose=10000) :
    my_rand = sample(range(N_fof),N_choose)
    pos_sampled = pos_fof[my_rand]
    #fig, ax = plt.figure(figsize=[10,10])
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=[10,10])
    plt.scatter(pos_sampled[:,ax_1],pos_sampled[:,ax_2],s=0.01,c='b')
    circle = plt.Circle((0,0),Rvir[h],color='r',fill=False)
    ax.add_patch(circle)
    my_min, my_max = np.min(pos_fof), np.max(pos_fof)
    ax.set_xlim(my_min,my_max)
    ax.set_ylim(my_min,my_max)
    if ax_2 == 1:
        ax_2_str = 'y'
    else :
        ax_2_str = 'z'
    ax.set_xlabel('x (Mpc/h)')
    ax.set_ylabel(ax_2_str+' (Mpc/h)')
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box') 
    path_plots = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/test_N_part_and_example/plots/'
    plt.savefig(path_plots+name+'_scatter_x_'+ax_2_str+'_full.pdf')
    plt.show()
    return()

if __name__ == '__main__':
    path_data = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/'
    path_fof = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_FOF/data/'
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/'
    path_plot = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/input_MF/data_halo/analytic_vs_DEUS/plots/relaxed_halos/'
    path_prop = path+'properties/relaxed_halos/'
    path_data = path_data+'/data/sphere_1_Rvir/'
    name_prop = 'prop_relaxed.dat' # contains Rvir, Nvir, cdm_fof, cdm_dens
    name_names = 'names_relaxed.dat' # contains halo names
    path_prop = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_beyond_FOF/halo_properties/cluster_csv/'
    prop = np.array(pd.read_csv(path_prop+'properties_numbers_all.dat'))
    Sph_all = np.array(pd.read_csv(path_prop+'sphericities_sphere_1_Rvir.dat'))
    Pass_all = np.array(pd.read_csv(path_prop+'Passage_rotation_sphere_1_Rvir.dat'))
    #indices = np.loadtxt(path_prop+'indices_non_shifted_haloes.dat')
    #indices = np.array(indices,dtype=int)
    #prop = prop[indices]
    Rvir = prop[:,0]
    cdm = prop[:,5:8]
    #Nvir = prop[:,1]
    #cdm_fof = prop[:,2:5]
    #cdm_dens = prop[:,5:]
    names_all = np.array(pd.read_csv(path_prop+'names_all.dat'))[:,0]
    #names_all = names_all[indices]
    print(len(prop))
    print(len(names_all))
    N_halos = 1 #150
    num = 206 # 96 #206 # 177 # 165 #231 #128 #31 #34 #246 # 215 #357 #269 # 335 #185 #160
    # 246, 231, 177
    # 206
    print(prop[num])
    k = 0
    abg, abg_el = np.zeros((N_halos,6)), np.zeros((N_halos,6))
    t0 = time.time()
    r_max = 1.1
    hsml = 5 * 10**(-3)
    res = 3*hsml
    res /= Rvir
    N_bin = 12
    for h in range(num,num+N_halos) :
        print(h)
        name = names_all[h]
        name_use = name.replace('.dat','_sphere_1_Rvir')
        #fof_boxlen648_n2048_lcdmw7_strct_00246_03208_sphere_1_Rvir.dat
        print(names_all[h])
        pos = np.array(pd.read_csv(path_data+name_use+'.dat'))
        N_vir = len(pos)
        pos_fof = np.loadtxt(path_fof+name)
        print(pos_fof)
        print(len(pos_fof))
        print(len(pos_fof)/7)
        N_fof = np.int(len(pos_fof)/7)
        print(N_fof)
        pos_fof = np.reshape(pos_fof,(7,N_fof))
        pos_fof = pos_fof[0:3]
        pos_fof = np.reshape(pos_fof,(N_fof,3))
        print(pos_fof)
        pos_fof *= L_box
        pos_fof -= cdm[h]
        ########################################################################
        do_scatter(pos,N_vir,name_use.replace('.dat',''),N_choose=N_vir) 
        do_scatter(pos_fof,N_fof,name.replace('.dat',''),N_choose=N_fof) 
        #print(pos)
        Sph = Sph_all[h]
        print(Sph)
        S, Q = Sph[3], Sph[4]
        pos /= Rvir[h]
        print(N_vir)
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        radius = np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
        r_log_bin, rho_log_bin, N_part_in_shell = profile_log_r_bin_hist(radius,r_min=res[h],N_bin=N_bin)
        ind_sph_complete =  np.where( (radius < 1) & (radius > res[h]) )[0]
        N_vir_sph = len(ind_sph_complete)
        ########################################################################
        Pass = Pass_all[k]
        Pass = Pass.reshape((3,3))
        x, y, z = rotate_position(pos,Pass)
        r_el = np.sqrt(np.power(x,2) + np.power(y/Q,2) + np.power(z/S,2))
        ind_el_complete = np.where( (r_el < 1) & (r_el > res[h]/S) )[0]
        r_log_bin_el, rho_log_bin_el, N_part_in_shell_el = profile_log_r_bin_hist(r_el[ind_el_complete],r_min=res[h]/S,r_max=1,N_bin=N_bin)
        N_vir_el = len(r_el[ind_el_complete])
        print(N_vir_sph,N_vir_el)
        def profile_abc(r_data,concentration,beta,gamma):
            alpha = 1
            # profile abc: rho(r) = rho_s * ( x**(-gamma) ) * (1 + x**alpha)**((gamma-beta)/alpha), x=r/r_s
            # r_data should be in Mpc/h
            # rho_s should be in rho_crit
            # concentration, alpha, beta, gamma are dimensionless
            mass = N_vir_sph #np.power(10,3) #rho_vir_Brian_Norman * (4*np.pi/3) * (Rvir**3) # in (mass_one_particle)
            Rvir = 1
            factor_dens = 1
            r_minus_2 = Rvir/concentration
            r_s = r_minus_2 * ((beta - 2)/(2 - gamma))**(1/alpha) # the scale radius in function of r_minus_2 in a alpha, beta, gamma profile
            prof_abc_x_2 = lambda x: (x**(2-gamma)) * (1 + x**alpha)**((gamma - beta)/alpha)
            my_int = integrate.quad(lambda x: prof_abc_x_2(x),0,Rvir/r_s)
            rho_s = (mass/(4*np.pi*(r_s**3)*my_int[0])) * factor_dens
            x_data = r_data/r_s # dimensionless
            rho = rho_s * ( x_data**(-gamma) ) * (1 + x_data**alpha)**((gamma-beta)/alpha) # in rho_crit
            return(np.log10(rho))
        
        my_bound_NFW = ([2,2.99,0.99],[20,3.01,1.01])
        popt_NFW, pcov_NFW = curve_fit(profile_abc,r_log_bin,np.log10(rho_log_bin),
                                       bounds=my_bound_NFW)
        perr = np.sqrt(np.diag(pcov_NFW))
        abg[k,0:3] = popt_NFW
        abg[k,3:] = perr
        rho_th = np.power(10,profile_abc(r_log_bin,popt_NFW[0],popt_NFW[1],popt_NFW[2]))
        my_bound_abg = ([2,2.1,0],[20,5,1.9])
        popt_abg, pcov_abg = curve_fit(profile_abc,r_log_bin,np.log10(rho_log_bin),
                                       bounds=my_bound_abg)
        rho_abg = np.power(10,profile_abc(r_log_bin,popt_abg[0],popt_abg[1],popt_abg[2]))
        
        popt_abg_el, pcov_abg_el = curve_fit(profile_abc,r_log_bin_el,np.log10(rho_log_bin_el),
                                             bounds=my_bound_NFW)
        perr_el = np.sqrt(np.diag(pcov_abg_el))
        my_chi_2 = np.sqrt(np.sum(((np.log10(rho_th) - np.log10(rho_log_bin))**2))/(N_bin-1))
        print('chi_2 sph =',np.log10(my_chi_2))
        abg_el[k,0:3] = popt_abg_el
        abg_el[k,3:] = perr_el
        rho_th_el = np.power(10,profile_abc(r_log_bin_el,popt_abg_el[0],popt_abg_el[1],popt_abg_el[2]))
        rho_abg_el = np.power(10,profile_abc(r_log_bin_el,3.18,3.80,1.11))
        #do_plot_profile(r_log_bin, rho_log_bin, rho_th, path_plot, name+'')
        do_plot_profile(r_log_bin,rho_log_bin,rho_abg,rho_th,name.replace('.dat',''))
        do_plot_profile_r_2(r_log_bin,rho_log_bin,rho_abg,rho_th,name.replace('.dat',''))
        #do_plot_profile(r_log_bin_el,rho_log_bin_el,rho_abg_el,rho_th_el,name.replace('.dat',''))
        #do_plot_profile_r_2(r_log_bin_el,rho_log_bin_el,rho_abg_el,rho_th_el,name.replace('.dat',''))
        k += 1
        rho_max = np.max(rho_log_bin*r_log_bin**2)
        ind_max = np.where(rho_log_bin*r_log_bin**2 == rho_max)[0]
        r_minus_2 = r_log_bin[ind_max] 
        my_c = 1/r_minus_2
        my_chi_2_el = np.sqrt(np.sum(((np.log10(rho_th_el) - np.log10(rho_log_bin_el))**2))/(N_bin-1))
        print('chi_2 el =',np.log10(my_chi_2_el))
        print(my_c)
        print(ind_max)
    t1 = time.time()
    print(t1-t0)
    print(abg)
    print(abg_el)
    abg = pd.DataFrame(abg,columns=['c','beta','gamma','c_err','beta_err','gamma_err'])
    #abg.to_csv(path_prop+'abg_non_shifted_haloes.dat')
    #np.savetxt(path_prop+'abg_cbg_Nbin_30_realxed.dat',abg)
    
'''
# Arturo
bins = np.logspace(login, logent, binnum) 
hist, bins = np.histogram(particle_radius, bins=bins) 
Vol = (4*np.pi/3)*(bins[1:]**3 -bins[:-1]**3) 
'''