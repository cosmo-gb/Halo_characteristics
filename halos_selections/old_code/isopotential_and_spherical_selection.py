#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:00:08 2021

@author: guillaume
"""


"""
This script performs a spherical and/or an isopotential selection of the data beyond the FOF algorithm.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
from matplotlib.ticker import MultipleLocator
import unsiotools.simulations.cfalcon as falcon
cf=falcon.CFalcon()
from parameters_DEUS_fof_boxlen648_n2048_lcdmw7 import hsml, v_ren, L_box, rho_vir_Brian_Norman, mass_one_particle, G_Newton

def periodicity(pos,cdm,dis_1=0.5,dis_2=0.9,box=1):
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
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
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
        pos = np.array((x,y,z)).transpose()
    return(pos,bad_periodicity)

def get_data(path,name,cdm) :
    # here I load the data beyond the FOF,
    # check the periodicity returns the data in the cdm frame in Mpc/h
    name_file_use = name+'.dat_5_Rvir_new.dat'
    data = np.array(pd.read_csv(path+name_file_use))
    pos = data[:,0:3]
    pos,bad_periodicity = periodicity(pos,cdm)
    print('bad periodicity =',bad_periodicity)
    pos *= L_box
    pos -= cdm*L_box
    return(pos)

def selection_sphere(pos,Rvir) :
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    rad = np.sqrt(x**2 + y**2 + z**2)
    ind_sphere = np.where(rad < Rvir)[0]
    pos_sphere = pos[ind_sphere]
    return(pos_sphere,ind_sphere)

def get_phi(pos,N_part,Rvir=1,mass_unit=1):
    pos_use = pos/Rvir
    pos_use = np.reshape(pos_use,N_part*3)
    pos_use = np.array(pos_use,dtype=np.float32)
    mass_array = np.ones((N_part),dtype=np.float32) * mass_unit
    ok, acc, pot = cf.getGravity(pos_use,mass_array,hsml)
    return(pot)

def plotting(pos,pot,my_title,name_saved,path_plot,ax_1=0,ax_2=1,size=0.01,
            x_min=-1,x_max=1,y_min=-1,y_max=1,
            x_label='x (Rvir)',y_label='y (Rvir)') :
    fig, ax = plt.subplots(figsize=[10,10])
    plt.ioff() # Turn interactive plotting off
    plt.rc('font', family='serif')
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    my_font = 20
    plt.rcParams['font.size'] = my_font
    ax.scatter(pos[:,ax_1],pos[:,ax_2],s=size,c=pot) #cmap='magma')
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    ax.tick_params(which='both', width=0.5, direction="in", 
                   labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    ax.tick_params(which='major', length=12) # I set the length
    ax.tick_params(which='minor', length=5)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(my_title,pad=20)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box') 
    #path = '../../../../DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_14_5_sphere_sel/plots/scatter_plots/iso_0_03_Rvir/'
    plt.savefig(path_plot+name_saved,format='pdf',transparent=True)
    plt.close(fig)
    #plt.show()
    return()

def do_plot(pos,pot,my_title,path_plot,N_sample=1000,unit=1,size=0.01,
            x_min=-1,x_max=1,y_min=-1,y_max=1, fac_renorm=1) : #x_label='x (Rvir)',y_label='y (Rvir') :
    x_label = 'x (Rvir/'+str(fac_renorm)+')'
    y_label = 'y (Rvir/'+str(fac_renorm)+')'
    z_label = 'z (Rvir/'+str(fac_renorm)+')'
    pos_use = pos/fac_renorm
    N_part_tot = len(pos_use)
    if N_part_tot > N_sample :
        my_ind = sample(range(N_part_tot),N_sample)
        pos_plot, pot_plot = pos_use[my_ind]/unit, pot[my_ind]
    else :
        pos_plot, pot_plot = pos_use/unit, pot
        
    plotting(pos_plot, pot_plot, my_title,my_title+'_x_y.pdf', path_plot,
             ax_1=0,ax_2=1,size=size,
             x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max,x_label=x_label,y_label=y_label)
    plotting(pos_plot, pot_plot, my_title, my_title+'_x_z.pdf', path_plot,
             ax_1=0,ax_2=2,size=size,
             x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max,x_label=x_label,y_label=z_label)
    return()

def sphere_Rvir(pos,Rvir,pot,fac_rad_max,path_data,name,add_name,path_plot,N_sample=10000,size=0.1) :
    # sphere selection
    pos_sphere, ind_sphere = selection_sphere(pos,fac_rad_max*Rvir)
    radius_sphere = np.sqrt(pos_sphere[:,0]**2 + pos_sphere[:,1]**2 + pos_sphere[:,2]**2)
    print('r_max (sphere '+str(fac_rad_max)+' Rvir) =',np.max(radius_sphere)/Rvir)
    #pos_sphere_Rvir = pd.DataFrame(pos_sphere_Rvir,columns=['in sphere of Rvir: x (Mpc/h) (in sphere of Rvir)','y (Mpc/h)','z (Mpc/h)'])
    #name_sphere = names_all[h].replace('.dat','_sphere_sel_Rvir.dat')
    title_sphere = name_file.replace('fof_boxlen648_n2048_lcdmw7_strct_',add_name+'_')
    do_plot(pos_sphere,np.abs(pot[ind_sphere]),
            title_sphere,path_plot+add_name+'/',
            N_sample=N_sample,unit=Rvir,size=size,fac_renorm=fac_rad_max)
    pos_sphere = pd.DataFrame(pos_sphere,columns=['x (Mpc/h)','y (Mpc/h)','z (Mpc/h)'])
    name_sphere = name.replace('.dat','_'+add_name+'.dat')
    path_and_name = path_data+add_name+'/'+name_sphere
    file = open(path_and_name, 'w')
    file.write('# Spherical selection  of '+str(fac_rad_max)+' Rvir of a halo. \n')
    pos_sphere.to_csv(file,index=False)
    file.close()
    return()

def iso_pot_beyond_Rvir(pos,Rvir,pot,name_file,
                        fac_rad_min=0.8,fac_rad_max=1,q_pot_threshold=0.1,N_sample=10000,size=0.1) :
    # isopotential beyond 1 Rvir
    radius = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    ind_shell = np.where( (radius < fac_rad_max * Rvir) & (radius > fac_rad_min * Rvir) )[0]
    pot_threshold = np.quantile(pot[ind_shell],q=q_pot_threshold)
    print('pot_threshold (iso beyond Rvir) =',pot_threshold)
    ind_pot = np.where(pot < pot_threshold)[0]
    ind_re_sphere = np.where(radius[ind_pot] < fac_rad_max * Rvir)[0]
    print('N_part_sel (iso beyond Rvir)= ',len(ind_re_sphere))
    ind_sel = ind_pot[ind_re_sphere]
    pos_sel = pos[ind_sel]
    print('r_max (pot beyond) =',np.max(radius[ind_sel])/Rvir)
    title_pot = name_file.replace('fof_boxlen648_n2048_lcdmw7_strct_','pot_beyond_')
    do_plot(pos_sel,np.abs(pot[ind_sel]),title_pot,
            N_sample=N_sample,unit=Rvir,size=size)
    #pos_sel = pd.DataFrame(pos_sel,columns=['isopotential 0.1 quantile in shell [0.8,1] Rvir, inside sphere Rvir: x (Mpc/h)','y (Mpc/h)','z (Mpc/h)'])
    #name_sel = names_all[h].replace('.dat','_isopotential_Rvir.dat')
    #pos_sel.to_csv(path_data+'isopotential_selection/'+name_sel,index=False)
    return()

def iso_not_beyond(pos,Rvir,Nvir,path_data,name,add_name,path_plot,
                   fac_rad_min=0.8,fac_rad_max=1,q_pot_threshold=0.5,
                   N_sample=10000,size=0.1) :
    # isopotential not beyond
    pos_sphere, ind_sphere = selection_sphere(pos,fac_rad_max*Rvir)
    radius_sphere = np.sqrt(pos_sphere[:,0]**2 + pos_sphere[:,1]**2 + pos_sphere[:,2]**2)
    N_part_sphere = len(pos_sphere)
    pot_sphere = get_phi(pos_sphere,N_part_sphere,Rvir,1/Nvir)
    ind_shell_sphere = np.where( (radius_sphere < fac_rad_max * Rvir) & (radius_sphere > fac_rad_min * Rvir) )[0]
    pot_threshold_sphere = np.quantile(pot_sphere[ind_shell_sphere],q=q_pot_threshold)
    print('pot_threshold (iso not beyond Rvir) =',pot_threshold_sphere)
    ind_pot_sphere = np.where(pot_sphere < pot_threshold_sphere)[0]
    pos_sel = pos_sphere[ind_pot_sphere]
    print('N_part_sel (iso not beyond Rvir)= ',len(pos_sel))
    r_max = np.max(radius_sphere[ind_pot_sphere])/Rvir
    print('r_max (pot not beyond) =',r_max)
    title_pot = name_file.replace('fof_boxlen648_n2048_lcdmw7_strct_',add_name+'_')
    do_plot(pos_sel,np.abs(pot_sphere[ind_pot_sphere]),title_pot,path_plot+add_name+'/',
            N_sample=N_sample,unit=Rvir,size=size,fac_renorm=fac_rad_max)
    pos_sel = pd.DataFrame(pos_sel,columns=['x (Mpc/h)','y (Mpc/h)','z (Mpc/h)'])
    name_sel = name.replace('.dat','_'+add_name+'.dat')
    path_and_name = path_data+add_name+'/'+name_sel
    file = open(path_and_name, 'w')
    file.write('# Isopotential selection of a halo. \n')
    file.write('# In a sphere of '+str(fac_rad_max)+' Rvir, the potential gravitational of each particle has been computed. \n')
    file.write('# Then a selection has been applied with phi_thresold = quantile(phi('+str(fac_rad_min)+'Rvir<r<'+str(fac_rad_max)+'Rvir),q='+str(q_pot_threshold)+'). \n')
    pos_sel.to_csv(file,index=False)
    file.close()
    issue = 0
    if r_max > 0.99*fac_rad_max :
        issue = 1
    return(np.array((issue,r_max)))

def save_r_max_array(r_max_array,path,add_name,fac_rad_max) :
    r_max_array = pd.DataFrame(r_max_array,columns=['issue (if no problem = 0)','r_max_sel (Rvir)'])
    file = open(path+'r_max_array_'+add_name+'.dat', 'w')
    file.write('# if a halo selection via isopotential has too large radii, \n')
    file.write('# it means that the selection has sphericized the halo. \n')
    file.write('# I compute the max(radius) and save it. issue=1 for r_max >= '+str(0.99*fac_rad_max)+' Rvir \n')
    r_max_array.to_csv(file,index=False)
    file.close()
    return()

cluster = True
mass_log = '12.5'
if cluster == True :
    path_use = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/'
    path_data = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/data/'
    path_prop = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/halo_properties/'
    path_fof = path_use+'/mass_10_'+mass_log+'_Msun_FOF/data/'
    path_out = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/'
    path_plot = '../../../../DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_12_5_sphere_sel/plots/scatter_plots/'
    names_all = np.array(pd.read_csv(path_prop+'names_all.dat'))[:,0]
    N_halos = len(names_all)
    #N_halos = 2
else :
    path_use = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/'
    path_data = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/data/'
    path_prop = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/halo_properties/cluster_csv/'
    path_fof = path_use+'/mass_10_'+mass_log+'_Msun_FOF/data/'
    path_out = path_use+'/mass_10_'+mass_log+'_Msun_beyond_FOF/'
    path_plot = '../../../../DEUSS_648Mpc_2048_particles/input_MF/stat_analytic_DEUS/mass_10_12_5_sphere_sel/plots/scatter_plots/'
    names_all = np.array(pd.read_csv(path_prop+'names_all.dat'))[:,0]
    N_halos = 2

prop_all = np.array(pd.read_csv(path_prop+'properties_numbers_all.dat'))
cdm_pot_fof = np.array(( prop_all[:,5], prop_all[:,6], prop_all[:,7])).transpose()
Rvir_all = prop_all[:,0]
Nvir_all = prop_all[:,1]
r_max_array = np.zeros((N_halos,2))
add_name_iso = 'iso_1_Rvir'
add_name_sphere = 'sphere_1_Rvir'
#path_plot = path_plot+add_name+'/'
fac_rad_min, fac_rad_max = 0, 1 #0.16, 0.2

if __name__ == '__main__':
    for h in range(N_halos) :
        Rvir, Nvir = Rvir_all[h], Nvir_all[h]
        print('Nvir =',Nvir)
        name_file = names_all[h].replace('.dat','')
        print(name_file)
        pos = get_data(path_data,name_file,cdm_pot_fof[h]/L_box)
        N_part_tot = len(pos)
        pot = get_phi(pos,N_part_tot,Rvir,1/Nvir)
        #######################################################################
        # sphere 1 Rvir
        sphere_Rvir(pos,Rvir,pot,fac_rad_max,path_data,names_all[h],add_name_sphere,path_plot) 
        #######################################################################
        # isopotential beyond Rvir
        #iso_pot_beyond_Rvir(pos,Rvir,pot,name_file,
        #                    fac_rad_min=0.8,fac_rad_max=1,N_sample=10000,size=0.1)
        #######################################################################
        # isopotential not beyond
        '''r_max_array[h] = iso_not_beyond(pos,Rvir,Nvir,path_data,names_all[h],add_name_iso,path_plot,
                                        fac_rad_min=fac_rad_min,fac_rad_max=fac_rad_max,q_pot_threshold=0.1)
    save_r_max_array(r_max_array,path_prop,add_name_iso,fac_rad_max)
'''