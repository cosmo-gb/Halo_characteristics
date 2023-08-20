#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:21:51 2022

@author: guillaume
"""


'''
This script contains the class where I generate smooth halo from density profiles
'''

import numpy as np
import sys
import random
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

import unsiotools.simulations.cfalcon as falcon
cf = falcon.CFalcon()

###############################################################################
from density_profile import Profile

class Smooth_halo(Profile):
    
    def around_integer_from_float(self,my_float):
        if isinstance(my_float,float) :    
            my_int = int(my_float)
            p = 1 - (my_float - my_int) # proba of adding +1 to the number of subhalos selected
            add = np.random.binomial(1, p)
            my_int += add
        elif isinstance(my_float,np.ndarray) :
            my_int = np.array(my_float,dtype=int)
            p = 1 - (my_float - my_int) # proba of adding +1 to the number of subhalos selected
            add = np.random.binomial(1, p)
            my_int += add
        else :
            print('Problem: my_float should be an integer or an array')
        return(my_int)
    
    def set_particles_log_slope(self,N_part_bin,R_min,R_max,alpha):
        r_cube_alpha_min, r_cube_alpha_max = np.power(R_min,3+alpha), np.power(R_max ,3+alpha)  
        r_cube_alpha = np.array(np.random.uniform(r_cube_alpha_min, r_cube_alpha_max,N_part_bin),dtype=np.float32)
        radius = np.power(r_cube_alpha,1/(3+alpha))
        my_theta_arcos = np.array(np.random.uniform(-1,1,N_part_bin),dtype=np.float32)
        phi = np.array(np.random.uniform(0,2*np.pi,N_part_bin),dtype=np.float32)
        return(radius,my_theta_arcos,phi)
    
    def change_coord_sphere_cart_ellipse_new(self,radius,my_theta_arcos,phi,a=1,b=1,c=1):
        ################################################################
        # this part becomes slow when N_part_bin becomes very large
        # I think I can not speed it up
        theta = np.arccos( my_theta_arcos )
        x, y, z = radius * np.sin(theta) * np.cos(phi) * a, radius * np.sin(theta) * np.sin(phi) * b, radius * np.cos(theta) * c
        ################################################################
        return(x,y,z)
    
    def throw_particles_log_slope(self,r_bin,N_part_bin,N_bin,log_slope,a_ax,b_ax,c_ax) :
        N_part_sum = np.cumsum(N_part_bin) # cumulative particle number,  contains N_bin + 1 elements
        N_tot = N_part_sum[-1] # total number of particle
        x, y, z = np.zeros((N_tot),dtype=np.float32), np.zeros((N_tot),dtype=np.float32), np.zeros((N_tot),dtype=np.float32)
        r_ell = np.zeros((N_tot),dtype=np.float32)
        for b in range(N_bin):
            beg, end = N_part_sum[b], N_part_sum[b+1]
            # I throw particles inside the shell b
            rad, my_theta_arcos, phi = self.set_particles_log_slope(end-beg,r_bin[b],r_bin[b+1],log_slope[b])
            x[beg:end], y[beg:end], z[beg:end] = self.change_coord_sphere_cart_ellipse_new(rad,my_theta_arcos,phi,a_ax[b],b_ax[b],c_ax[b])
            S, Q = c_ax[b]/a_ax[b], b_ax[b]/a_ax[b]
            r_ell[beg:end] = np.sqrt( x[beg:end]**2 + (y[beg:end]/Q)**2 + (z[beg:end]/S)**2 )
        return(x,y,z,r_ell,N_tot)
    
    def plot_data(self,x,y,x_min=-1,x_max=1,y_min=-1,y_max=1) :
        plt.rc('font', family='serif')
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots() # plot
        ax.tick_params(which='both', width=0.5, direction="in",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        #####################################################################################""
        ax.scatter(x,y,c='r',s=0.001)
        ########################################################################################
        #ax.set_xlim(np.min(x),np.max())
        #ax.set_ylim(y_min,y_max)
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        return()

    def beauty_plot_colorbar(self,pos,path_and_name='',pixels=1000,r_max=1,ax_1=0,ax_2=1,) :
        ############################################################################################
        N_part = len(pos)
        mass_array = np.ones((N_part),dtype=np.float32)
        pos_use = np.zeros((N_part,3),dtype=np.float32) 
        pos_use[:,0], pos_use[:,1], pos_use[:,2] = pos[:,0], pos[:,1], pos[:,2]
        pos_use = np.reshape(pos_use,N_part*3)
        hsml = 0.01
        ok, acc, pot = cf.getGravity(pos_use,mass_array,hsml)
        mass = -pot
        ###############################################################################""""
        plt.rc('font', family='serif',size=20)
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
        fig, ax = plt.subplots(figsize=[10,10]) # plot
        ax.tick_params(which='both', width=1, direction="out",
                       labeltop=False,labelbottom=True, labelleft=True, labelright=False)
        ###########################################################################################
        # pixels = 1000
        # limits for the data you want to consider for the histogram
        ax.set_xlim([-r_max,r_max])
        ax.set_ylim([-r_max,r_max])
        # this will be a squared plot so I define the same array for x and y
        edges = np.linspace(-r_max,r_max,pixels)
        # this is a way of make it easier if you want to see a different projection than the plan x-y
        # this is the histogram one by numpy 
        H, xedges, yedges = np.histogram2d(pos[:,ax_1],# the data in the x axis 
                                       pos[:,ax_2],# the data for the y axis
                                       bins=(edges, edges), # the bins for boths sets of data 
                                       weights=mass# here you wright your data it can be any array as long as it has the same length as x and y
                                      )
        # this is stupid but is numpy's fault you need to take the transposed of the matrix that represent the 2dhistogram
        fullbox = H.T
        # this is arbitrary you will see
        a,b = np.mean(fullbox)*0.05, np.max(fullbox)/2 # Arturo's guess
        # now we plot
        mass_2 = ax.imshow(fullbox+0.1,# the added number is to have all bins with color
                       interpolation='gaussian',# this defines how you plot frontiers between adjacent bins
                       origin='lower', # if you dont do this you get a weird orientation on the plot
                       cmap="magma", # here is your choice for the color map
                       extent=[edges[0], edges[-1], edges[0], edges[-1]], # the limits
                       norm=LogNorm(vmin=a,vmax=b), # this is a trick, you have to import "from matplotlib.colors import LogNorm" so here you use the values I defined as limits but is by playin that i know this|
                      )
        plt.subplots_adjust(bottom=0., right=0.8, top=1)
        cax = plt.axes([0.85, 0.15, 0.075, 0.7])
        plt.colorbar(mass_2,cax=cax)
        ax.set_xlabel('x ($R_{vir}$)')
        ax.set_ylabel('y ($R_{vir}$)',labelpad=0)
        circle1 = plt.Circle((0, 0), 1, color='white',lw=2,fill=False)
        ax.add_patch(circle1)
        ax.text(0.7,-0.8,'1 Rvir',c='white')
        ################################################################################# 
        ax.tick_params(which='major', length=12) # I set the length
        ax.tick_params(which='minor', length=5)       
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
        ax.text(1.2*r_max,1.1*r_max,'$|\psi|$',c='k')
        if path_and_name != '' :
            plt.savefig(path_and_name)
        plt.show()
        return()
    
    def smooth_halo_creation(self, kind_profile,
                             N_part=10000, N_bin=30, R_min=0, R_max=1,
                             res=0.01, r_shell_binning='logarithmic', a_ax=1, b_ax=1, c_ax=1,):
        ''' 
        This is the main fuction of this program, it generates smooth halos at position x,y,z = 0,0,0
        It creates particles in a halo of a given analytical density profile.
        Parameters:
        -kind_profile should be a list containing the kind of profile you want.
        its first element is a string: either abg, Einasto, or single (for single slope)
        in the case of abg or Einasto, the concentration should also be given as a second parameter of the list,
        while in the case of single, the logarithmic slope delta shoulb be given
        -N_part is the total number of particles contained in the halo at the end
        -N_bin is the number of bins for the shells
        -R_min and R_max are the minimum and maximum radius of the halo
        -res is the resolution i.e. the size of the firs shell
        -r_shell_binning: the radius can be binned linearly but by default it is binned logarithmically
        -a_ax, b_ax, c_ax: the halo shape is given by a_ax, b_ax, c_ax (dimensionless so between 0 and 1),
        which correspond to the 3 mains axis
        it can be either float numbers (constant shape) or numpy arrays (shape varyng with the radius)
        This function returns:
        -data: which are the particle positions in the same unit as R_min and R_max
        -N_tot: the total number of particles N_tot, 
        -N_part_bin: the number of particles in each bin N_part_bin,
        -r_bin: the radius of each bin (or shell) r_bin
        -r: the radius of each particle r
        -r_ell: and the ellipsoidal radius of each particle r_ell
        '''
        ####################################################################### deal with x: the radius of the shells
        if r_shell_binning == 'logarithmic' : # shell bining with logarithmic radius bin
            if R_min != 0 : # radius limiting case, so all particles will have R_min < r < R_max
                r_bin = 10**(np.linspace(np.log10(R_min), np.log10(R_max), N_bin+1))
            else : # particles can be everywhere with r < R_max
                r_bin = np.logspace(np.log10(res),np.log10(R_max), N_bin)
                r_bin = np.append(np.array((0)), r_bin)
        else : # shell bining with linear radius bin
            if R_min != 0 : # radius limiting case, so all particles will have R_min < r < R_max
                r_bin = np.linspace(R_min, R_max, N_bin+1)
            else : # particles can be everywhere with r < R_max
                r_bin = np.linspace(R_max/N_bin, R_max, N_bin)
                r_bin = np.append(np.array((0)), r_bin) 
        ####################################################################### deal with sphericity:
        if isinstance(c_ax, (np.ndarray)) : # it means that it is an array, local shape, change with the radius
             if len(c_ax) != N_bin or len(b_ax) != N_bin or len(a_ax) != N_bin :
                 sys.exit('the shape abc should be either floats or arrays of the size of N_bin=',N_bin)
        else : # it means that it is a number, global shape everywhere in the halo
            # thus I transform that shape in constant array
            a_ax = np.ones((N_bin)) * a_ax
            b_ax = np.ones((N_bin)) * b_ax
            c_ax = np.ones((N_bin)) * c_ax
        ####################################################################### deal with kind_profile:
        r_s, n_x, log_slope = self.deal_with_kind_profile(kind_profile, r_bin, R_max)
        n_x_2 = lambda x: (x**(2)) * n_x(x)
        ####################################################################### set everything in r_s unit:
        R_min, R_max, res = R_min/r_s, R_max/r_s, res/r_s
        r_bin = r_bin/r_s
        ####################################################################### total number of particles in the halo and normalisation of the density profile
        my_int_tot = integrate.quad(lambda x: n_x_2(x), R_min, R_max) # volume in r_s**3 unit
        n_s = N_part/(4*np.pi*my_int_tot[0]) # scale number density in r_s**(-3) unit
        ####################################################################### deal with y: number of particles in each shell
        N_part_bin = np.zeros((N_bin+1),dtype=int) # it contains N_bin + 1 elements
        proba = np.zeros((N_bin),dtype=int) # it contains N_bin elements
        for b in range(N_bin):
            # I compute the number of particles in each radius bin i.e. in each shell
            my_int =  integrate.quad(lambda r: n_x_2(r), r_bin[b], r_bin[b+1]) # this is in mass unit
            #N_part_bin[b+1] = self.around_integer_from_float( (4*np.pi*n_s) * my_int[0] ) # number of particle in the bin b
            #N_part_bin[b+1] = int(np.round((4*np.pi*n_s) * my_int[0], decimals=0)) # number of particle in the bin b
            N_part_float = (4*np.pi*n_s) * my_int[0]
            N_part_int = int(N_part_float)
            N_part_bin[b+1] = N_part_int
            proba[b] = 1 - (N_part_float - N_part_int) # proba of NOT set a particle in this bin b
        ####################################################################### throw the particles inside the shells according to the density
        x, y, z, r_ell, N_tot = self.throw_particles_log_slope(r_bin, N_part_bin, N_bin,  log_slope, a_ax, b_ax, c_ax)
        ###################################################################################################################"
        # create N_part_bin_bonus:
        ind_p = np.argsort(proba) # sorted proba, large proba of set a particle first
        N_part_bin_bon = np.zeros((N_bin),dtype=int)
        N_miss = N_part - N_tot
        for miss in range(N_miss) : # this could be more elegant
            N_part_bin_bon[ind_p[miss]] = 1
        N_part_bin_bon = np.append(0,N_part_bin_bon)
        if N_miss > 0 and N_miss <= N_bin :
            x_bon, y_bon, z_bon, r_ell_bon, N_tot_bon = self.throw_particles_log_slope(r_bin, N_part_bin_bon, N_bin,  log_slope, a_ax, b_ax, c_ax)
            x, y, z = np.append(x,x_bon), np.append(y,y_bon), np.append(z,z_bon)
            r_ell, N_tot = np.append(r_ell,r_ell_bon), N_tot + N_tot_bon
        else :
            print('Problem: N_miss=',N_miss)
        r = np.sqrt(x**2 + y**2 + z**2) # radius of each particle in the halo
        data = np.zeros((N_tot,3), dtype=np.float32)
        data[:,0], data[:,1], data[:,2] = x, y, z # particle positions in the same unit as R_max
        ####################################################################### finally I reset every length in R_max unit
        data = data * r_s
        r_bin, r, r_ell = r_bin * r_s, r * r_s, r_ell * r_s
        n_s = n_s/r_s**3
        return(data, N_tot, N_part_bin, r_bin, r, r_ell, r_s, n_s, n_x)
    
    def do_many_times(self, N_times,):
        c = 10
        kind_profile = ['abg',c]
        N_tot_th = 10000
        N_tot_data = np.zeros((N_times))
        for n in range(N_times) :    
            my_halo = self.smooth_halo_creation(kind_profile,N_part=N_tot_th)
            N_tot_data[n] = my_halo[1]
        print('N_tot_mean =',np.mean(N_tot_data))
        print('N_tot_std =',np.std(N_tot_data))
        return()
    
    
if __name__ == '__main__':
    halo = Smooth_halo()
    c = 10
    kind_profile = ['abg', c]
    my_halo = halo.smooth_halo_creation(kind_profile, b_ax=0.5, c_ax=0.5)
    data = my_halo[0]
    print('N_tot =',my_halo[1])
    #halo.plot_data(data[:,0],data[:,1])
    halo.beauty_plot_colorbar(data)
    #halo.do_many_times(10)


