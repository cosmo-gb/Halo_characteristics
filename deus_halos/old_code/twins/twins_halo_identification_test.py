#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:29:02 2021

@author: guillaume
"""

'''
Compare distances between haloes from different simulations
Identify if haloes are the same but in different cosmology
'''

import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def compute_dis(pos_1,pos_2,Rvir_1):
    N_1 = len(pos_1)
    print('N_1 =',N_1)
    N_2 = len(pos_2)
    print('N_2 =',N_2)
    # indices of the population 2 corresponding to the population 1
    ind_h2 = np.zeros(N_1,dtype=int)
    dis_h2 = np.zeros(N_1)
    for h1 in range(N_1) :
        dis_h1 = np.sqrt( (pos_1[h1,0]-pos_2[:,0])**2 + (pos_1[h1,1]-pos_2[:,1])**2 + (pos_1[h1,2]-pos_2[:,2])**2 )
        dis_h1 /= Rvir_1[h1]
        dis_h1_min = np.min(dis_h1)
        ind_h2[h1] = np.where(dis_h1 == dis_h1_min)[0][0]
        dis_h2[h1] = dis_h1_min
    return(ind_h2,dis_h2)

def check_id_particles(id_1,id_2,N_s_1=10) :
    N_1, N_2 = len(id_1), len(id_2)
    ind_s_1 = sample(range(N_1),N_s_1)
    id_s_1 = id_1[ind_s_1]
    i_1 = 0
    for i in id_s_1 :
        if len(np.where(id_2 == i)[0]) == 1 :
            i_1 += 1
    return(i_1,i_1/N_s_1)

def select_frac_data(data_1,data_2,data_3,cdm_1,cdm_2,cdm_3,R_1,R_2,R_3) :
    L_box = 648
    pos_1 = data_1[:,0:3] * L_box
    pos_2 = data_2[:,0:3] * L_box
    pos_3 = data_3[:,0:3] * L_box
    print(pos_1)
    print(cdm_1)
    pos_1 = pos_1 - cdm_1
    pos_2 = pos_2 - cdm_2
    pos_3 = pos_3 - cdm_3
    r_1 = np.power(np.sum(pos_1**2,axis=1),0.5)
    r_2 = np.sqrt(np.sum(pos_2**2,axis=1))
    r_3 = np.sqrt(np.sum(pos_3**2,axis=1))
    ind_1 = np.where(r_1 < R_1)[0]
    ind_2 = np.where(r_2 < R_2)[0]
    ind_3 = np.where(r_3 < R_3)[0]
    id_1 = data_1[ind_1,-1]
    id_2 = data_2[ind_2,-1]
    id_3 = data_3[ind_3,-1]
    return(id_1,id_2,id_3,ind_1,ind_2,ind_3)


def scatter_plot(pos_1,pos_2,pos_3,N_s,ind_1=0,ind_2=1,s=0.1) :
    fig, ax = plt.subplots(figsize=[10,10])
    plt.ioff() # Turn interactive plotting off
    plt.rc('font', family='serif')
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
    my_font = 20
    plt.rcParams['font.size'] = my_font
    N_1 = len(pos_1)
    ind_s_1 = sample(range(N_1),N_s)
    ax.scatter(pos_1[ind_s_1,ind_1],pos_1[ind_s_1,ind_2],c='g',s=s,label='$\Lambda$CDM')
    N_2 = len(pos_2)
    ind_s_2 = sample(range(N_2),N_s)
    ax.scatter(pos_2[ind_s_2,ind_1],pos_2[ind_s_2,ind_2],c='r',s=s,label='RPCDM')
    N_3 = len(pos_3)
    ind_s_3 = sample(range(N_3),N_s)
    ax.scatter(pos_3[ind_s_3,ind_1],pos_3[ind_s_3,ind_2],c='b',s=s,label='wCDM')
    #circle1 = plt.Circle((0, 0), 0.2, color='r')
    #ax.set_xlim([37,48])
    #ax.set_ylim([90,102])
    ax.tick_params(which='both', width=0.5, direction="in", 
                   labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    ax.tick_params(which='major', length=12) # I set the length
    ax.tick_params(which='minor', length=5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_xlabel('x (Mpc/h)')
    ax.set_ylabel('y (Mpc/h)')
    ax.set_title('twin halo in DEUS',pad=20)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box') 
    plt.legend()
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/plots/scatter/'
    plt.savefig(path+'scatter_plot_cube_5_Rvir_3_cosmo_1_halo',format='pdf',transparent=True)
    #plt.close(fig)
    plt.show()
    return()
    
    
    
            


if __name__ == '__main__':
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'
    path_prop = 'mass_10_14.5_Msun_beyond_FOF/halo_properties/'
    cosmo = 'RPCDM'
    prop_name = 'properties_numbers_all.dat'
    prop_RP = np.array(pd.read_csv(path+'RPCDM/'+path_prop+prop_name))
    prop_w = np.array(pd.read_csv(path+'wCDM/'+path_prop+prop_name))
    print(prop_RP)
    print(prop_w)
    cdm_RP = prop_RP[:,5:8]
    Rvir_RP = prop_RP[:,0]
    cdm_w = prop_w[:,5:8]
    Rvir_w = prop_w[:,0]
    ind_w, dis_w = compute_dis(cdm_RP,cdm_w,Rvir_RP)
    print(ind_w, dis_w)
    print(np.min(dis_w))
    ind_RP = np.where(dis_w == np.min(dis_w))[0]
    print(ind_RP)
    print(ind_w[ind_RP])
    ind_w = ind_w[ind_RP]
    print(cdm_RP[ind_RP])
    print(cdm_w[ind_w])
    prop_name = 'properties_all.dat'
    prop_RP = np.array(pd.read_csv(path+'RPCDM/'+path_prop+prop_name))
    print(prop_RP[ind_RP])
    name_RP = prop_RP[ind_RP[0]][0]
    name_RP = name_RP.replace('.dat','_5_Rvir.dat')
    prop_w = np.array(pd.read_csv(path+'wCDM/'+path_prop+prop_name))
    name_w = prop_w[ind_w[0]][0]
    name_w = name_w.replace('.dat','_5_Rvir.dat')
    print(prop_w[ind_w])
    prop_L = np.array(pd.read_csv(path+'LCDM/'+path_prop+prop_name))
    cdm_L = prop_L[:,6:9]
    Rvir_L = prop_L[:,1]
    #print((cdm_L-cdm_RP[ind_best])**2)
    #print(np.sum((cdm_L-cdm_RP[ind_best])**2,axis=1))
    dis_2_L = np.sum((cdm_L-cdm_RP[ind_RP])**2,axis=1)
    #dis_cdm_L = np.sqrt(np.sum((cdm_L-cdm_RP[ind_best])**2,axis=1))
    #print(dis_2_L)
    #print(np.min(dis_2_L))
    ind_L = np.where(dis_2_L == np.min(dis_2_L))[0]
    print(ind_L)
    print(prop_L[ind_L])
    name_L = prop_L[ind_L[0]][0]
    name_L = name_L.replace('.dat','.dat_5_Rvir_new.dat')
    ######################################################################
    path_data = 'mass_10_14.5_Msun_beyond_FOF/data/'
    data_RP = np.array(pd.read_csv(path+'RPCDM/'+path_data+name_RP))
    #print(data_RP)
    id_RP = data_RP[:,-1]
    #print(id_RP)
    data_w = np.array(pd.read_csv(path+'wCDM/'+path_data+name_w))
    #print(data_w)
    id_w = data_w[:,-1]
    i_RP, f_RP = check_id_particles(id_RP,id_w,N_s_1=10)
    print(i_RP, f_RP)
    
    data_L = np.array(pd.read_csv(path+'LCDM/'+path_data+name_L))
    id_L = data_L[:,-1]
    i_L, f_L = check_id_particles(id_L,id_w,N_s_1=10)
    print(i_L, f_L)
    
    frac = 5
    N_s = 10
    id_L,id_RP,id_w, ind_L, ind_RP, ind_w = select_frac_data(data_L,data_RP,data_w,
                                      cdm_L[ind_L],cdm_RP[ind_RP],cdm_w[ind_w],
                                      frac*Rvir_L[ind_L[0]],frac*Rvir_RP[ind_RP[0]],frac*Rvir_w[ind_w[0]])
    print(check_id_particles(id_RP,id_L,N_s_1=N_s))
    print(check_id_particles(id_RP,id_w,N_s_1=N_s))
    
    L_box = 648
    pos_L, pos_RP, pos_w = data_L[:,0:3]*L_box,data_RP[:,0:3]*L_box,data_w[:,0:3]*L_box
    #pos_L, pos_RP, pos_w = data_L[ind_L,0:3]*L_box,data_RP[ind_RP,0:3]*L_box,data_w[ind_w,0:3]*L_box
    scatter_plot(pos_L, pos_RP, pos_w,N_s=5000)
    
    