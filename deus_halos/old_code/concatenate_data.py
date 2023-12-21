#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:11:46 2021

@author: guillaume
"""

""" This script concatenates the files resulting from FOF and beyond FOF haloes properties
and also from halo positions (get from the halo files of DEUS).
"""




import numpy as np
import pandas as pd
from random import sample
import os.path



def concatenate_data(FOF=False,redshift_str='0',cosmo='lcdm') :
    #path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_FOF/halo_properties_csv/'
    #path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_12.5_Msun_beyond_FOF/halo_properties/cluster_csv/'
    mass_str = '13'
    #cosmo = 'wcdm'
    if FOF == True :
        FOF_str = 'FOF'
        path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+mass_str+'.5_Msun_'+FOF_str+'/halo_properties/'
        N_col = 18
        file_name = 'haloes_usefull_' # FOF
        properties_names = 'properties_' # FOF
        properties = 'properties_numbers_' # FOF
    elif FOF == False :
        FOF_str = 'beyond_FOF'
        #path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+mass_str+'.5_Msun_'+FOF_str+'/halo_properties/'
        N_col = 11
        mass_name = 'R_and_M_saved_'
        file_name = 'names_saved_' # beyond FOF
        properties_names = 'prop_names_saved_' #  beyond FOF
        properties = 'prop_saved_' # beyond FOF 
    elif FOF == 'most_massive' :
        FOF_str = FOF
        path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+redshift_str+'/'+cosmo+'/most_massive/halo_properties/'
        N_col = 17
        mass_name = 'R_and_M_saved_'
        file_name = 'names_saved_' # beyond FOF
        properties_names = 'prop_names_saved_' #  beyond FOF
        properties = 'prop_saved_' # beyond FOF 
    # numbers for LCDM7 at z ~ 0: #99743 #1128175 #10280 # 374
    # wCDM7, z ~ 0, Mvir=[10^14.5,10^14.7]Msun/h, FOF, N_halos = 11199
    # wCDM7, z ~ 0, Mvir=[10^14.5,10^14.7]Msun/h, beyond FOF, N_halos = 556
    # RPCDM7, z ~ 0, Mvir=[10^14.5,10^14.7]Msun/h, FOF, N_halos = 3886
    # RPCDM7, z ~ 0, Mvir=[10^14.5,10^14.7]Msun/h, beyond FOF, N_halos = 83
    # RPCDM7, z ~ 0, Mvir=[10^13.5,10^13.7]Msun/h, FOF, N_halos = 203044
    # RPCDM7, z ~ 0, Mvir=[10^12.5,10^12.7]Msun/h, FOF, N_halos = 1681279
    # wCDM7, z ~ 0, Mvir=[10^13.5,10^13.7]Msun/h, FOF, N_halos = 296951
    # wCDM7, z ~ 0, Mvir=[10^12.5,10^12.7]Msun/h, FOF, N_halos = 1736934
    # RPCDM7, z ~ 0, Mvir=[10^13.5,10^13.7]Msun/h, beyond FOF, I selected randomly N_halos = 511
    # RPCDM7, z ~ 0, Mvir=[10^12.5,10^12.7]Msun/h, beyond FOF, I selected randomly N_halos = 510
    # wCDM7, z ~ 0, Mvir=[10^12.5,10^12.7]Msun/h, beyond FOF, I selected randomly N_halos = 510
    # wCDM7, z ~ 0, Mvir=[10^13.5,10^13.7]Msun/h, beyond FOF, I selected randomly N_halos = 509
    N_haloes = 10000
    names, prop, mass = np.array(()), np.zeros((N_haloes,N_col)), np.zeros((N_haloes,13))
    prop_name_saved = []
    N_in_loop_before, N_in_loop_after = 0, 0
    for i in range(1,129) :
        #if i != 56 and i!= 77 :
        #if i!=130  :
        print(i)
        if os.path.isfile(path+file_name+str(i)+'.dat') == True :
            names_saved = np.array(pd.read_csv(path+file_name+str(i)+'.dat')) #[:,0]
            #names_saved_1.dat
            #/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_12.5_Msun_beyond_FOF/halo_properties/cluster_csv
            N_in_loop_after += len(names_saved)
            names = np.append(names,names_saved)
            if FOF == False or FOF == 'most_massive' :
                mass_saved = np.array(pd.read_csv(path+mass_name+str(i)+'.dat'))
                mass[N_in_loop_before:N_in_loop_after] = mass_saved
            prop_saved = np.array(pd.read_csv(path+properties+str(i)+'.dat')) #[:,1:]
            prop_name_saved += list(np.array(pd.read_csv(path+properties_names+str(i)+'.dat'))) #[:,1:])
            #prop_names = np.array(pd.read_csv(path+properties_names+str(i)+'.dat'))[:,1:]
            #print(names)
            #print(prop_saved)
            prop[N_in_loop_before:N_in_loop_after] = prop_saved
            #prop = np.append(prop,np.array(pd.read_csv(path+properties+str(i)+'.dat')),axis=0)
            N_in_loop_before += len(names_saved)
    print(names)
    print(len(names))
    #print(prop)
    print(mass)
    print(len(prop))
    print(len(prop_name_saved))
    for_save = '10000 '+FOF_str+' halos name from beyond FOF data, '+cosmo+', z ~ '+redshift_str
    names = pd.DataFrame(names,columns=[for_save])
    names.to_csv(path+'names_all.dat',index=False)
    if FOF == True :
        # N_part_FOF,N_part_vir,Rvir,cdm_fof_x (Mpc/h),cdm_fof_y,cdm_fof_z,cdm_pot_x (Mpc/h),cdm_pot_y,cdm_pot_z,
        # cdm_dens_x (Mpc/h),cdm_dens_y,cdm_dens_z,cdm_shrink_x (Mpc/h),cdm_shrink_y,cdm_shrink_z,
        # shift s= |r_cpot -r_cfof|/Rvir,|r_cpot-r_cdens|,|r_cpot-r_cshrink|
        prop = pd.DataFrame(prop,columns=['NFOF','Nvir','Rvir (Mpc/h)',
                                          'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                          'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                          'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                          'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                          'shift s= |r_cpot-r_cfof|/Rvir','|r_cpot-r_cdens|','|r_cpot-r_cshrink|'])
        prop.to_csv(path+'properties_numbers_all.dat',index=False)
        prop_name_saved = pd.DataFrame(prop_name_saved,columns=['halo name','NFOF','Nvir','Rvir (Mpc/h)',
                                                                'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                                'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                                'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                                                'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                                                'shift s= |r_cpot-r_cfof|/Rvir','|r_cpot-r_cdens|','|r_cpot-r_cshrink|'])
        prop_name_saved.to_csv(path+'properties_all.dat',index=False)
    elif FOF == False :
        mass = pd.DataFrame(mass,columns=['Rvir (Mpc/h)','Nvir','R200m','N200m','R200c','N200c',
                                          'R500c','N500c','R1000c','N1000c','R2000c','N2000c','NFOF'])
        mass.to_csv(path+'R_and_M_all.dat',index=False)
        prop = pd.DataFrame(prop,columns=['Nvir','Rvir (Mpc/h)',
                                          'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                          'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                          'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z'])
        prop.to_csv(path+'properties_numbers_all.dat',index=False)
        prop_name_saved = pd.DataFrame(prop_name_saved,columns=['halo name','Nvir','Rvir (Mpc/h)',
                                                            'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                            'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                            'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z'])
        prop_name_saved.to_csv(path+'properties_all.dat',index=False)
    elif FOF == 'most_massive' :
        mass = pd.DataFrame(mass,columns=['Rvir (Mpc/h)','Nvir','R200m','N200m','R200c','N200c',
                                          'R500c','N500c','R1000c','N1000c','R2000c','N2000c','NFOF'])
        mass.to_csv(path+'R_and_M_all.dat',index=False)
        prop = pd.DataFrame(prop,columns=['Rvir (Mpc/h)','Nvir',
                                          'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                          'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                          'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                          'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                          'cdm_x (Mpc/h)','cdm_y','cdm_z'])
        prop.to_csv(path+'properties_numbers_all.dat',index=False)
        
        prop_name_saved = pd.DataFrame(prop_name_saved,columns=['halo name','Rvir (Mpc/h)','Nvir',
                                                                'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                                'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                                'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                                                'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                                                'cdm_x (Mpc/h)','cdm_y','cdm_z'])
        prop_name_saved.to_csv(path+'properties_all.dat',index=False)
    return()

def concatenate_halos(path,N_halos) :
    N_col = 7
    N_cubes = 512
    N_in_loop_before, N_in_loop_after = 0, 0
    prop_saved = np.zeros((N_halos,N_col))
    ide_saved = np.zeros((N_halos,3),dtype=int)
    for c in range(N_cubes) :
        print(c)
        prop = np.array(pd.read_csv(path+'halo_pos_'+str(c)+'.dat'))
        ide = np.array(pd.read_csv(path+'halo_id_'+str(c)+'.dat'),dtype=int)[:,0]
        N_in_loop_after += len(prop)
        N_halos_in_c = N_in_loop_after-N_in_loop_before
        prop_saved[N_in_loop_before:N_in_loop_after,0:5] = prop
        prop_saved[N_in_loop_before:N_in_loop_after,5] = c * np.ones((N_halos_in_c))
        prop_saved[N_in_loop_before:N_in_loop_after,6] = range(N_halos_in_c)
        ide_saved[N_in_loop_before:N_in_loop_after,0] = ide
        ide_saved[N_in_loop_before:N_in_loop_after,1] = c * np.ones((N_halos_in_c))
        ide_saved[N_in_loop_before:N_in_loop_after,2] = range(N_halos_in_c)
        N_in_loop_before = N_in_loop_after
    print(N_in_loop_after)
    prop_saved = pd.DataFrame(prop_saved,columns=['halo identity','halo N_part','x (Mpc/h)','y','z','cube c','halo numero'])
    prop_saved.to_csv(path+'halo_pos_all.dat',index=False)
    ide_saved = pd.DataFrame(ide_saved,columns=['halo identity saved as integer','cube c','halo numero'])
    ide_saved.to_csv(path+'halo_id_all.dat',index=False)
    return()
        
def sel_random_halos(mass_str='13',cosmo='rpcdm',N_sel=256):
    path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/z_0/'+cosmo+'/mass_10_'+mass_str+'.5_Msun_beyond_FOF/halo_properties/'
    names_full = np.array(pd.read_csv(path+'names_all.dat'))[:,0]
    prop_full = np.array(pd.read_csv(path+'properties_numbers_all.dat'))
    prop_names_full = np.array(pd.read_csv(path+'properties_all.dat'))
    mass_full = np.array(pd.read_csv(path+'R_and_M_all.dat'))
    N_halos = len(names_full)
    print('N_halos =',N_halos)
    my_int = sample(range(N_halos),N_sel)
    my_int_sel = np.sort(my_int)
    print(my_int_sel)
    int_save = pd.DataFrame(my_int_sel,columns=['integers of the random selection applied for Mvir = [10^'+mass_str+'.5,10^'+mass_str+'.7] Msun/h obtained from beyond FOF data, '+cosmo+', z ~ 0'])
    int_save.to_csv(path+'integers_random_sel_all.dat',index=False)
    names_sel = names_full[my_int_sel]
    prop_sel = prop_full[my_int_sel]
    prop_names_sel = prop_names_full[my_int_sel]
    mass_sel = mass_full[my_int_sel]
    print(names_sel[0:3])
    print(prop_names_sel[0:3])
    print(prop_sel[0:3])
    print(mass_sel[0:3])
    names = pd.DataFrame(names_sel,columns=[str(N_sel)+' halo names (random selection)  with Mvir = [10^'+mass_str+'.5,10^'+mass_str+'.7] Msun/h obtained from beyond FOF data, '+cosmo+', z ~ 0'])
    names.to_csv(path+'names_sel.dat',index=False)
    prop = pd.DataFrame(prop_sel,columns=['Rvir (Mpc/h)','Nvir',
                                      'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                      'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                      'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z'])
    prop.to_csv(path+'properties_numbers_sel.dat',index=False)
    prop_name_saved = pd.DataFrame(prop_names_sel,columns=['halo name','Rvir (Mpc/h)','Nvir',
                                                            'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                            'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                            'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z'])
    prop_name_saved.to_csv(path+'properties_sel.dat',index=False)
    mass = pd.DataFrame(mass_sel,columns=['Rvir (Mpc/h)','Nvir','R200m','N200m','R200c','N200c',
                                          'R500c','N500c','R1000c','N1000c','R2000c','N2000c','NFOF'])
    mass.to_csv(path+'R_and_M_sel.dat',index=False)
    return()

def sel_most_massive(cosmo,redshift,N_halos) :
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+redshift+'/'+cosmo+'cdm/halos_position/halo_properties/'
    prop_all = np.array(pd.read_csv(path+'halo_pos_all.dat'))
    N_part_all = prop_all[:,1]
    ind_sort = np.argsort(N_part_all)
    ind_pre_sel = ind_sort[-N_halos:]
    prop_pre_sel = prop_all[ind_pre_sel]
    #prop_saved = pd.DataFrame(prop_pre_sel,columns=['halo identity','halo N_part','x (Mpc/h)','y','z'])
    file_save = open(path+'halo_pos_pre_sel.dat', 'w')
    file_save.write('# '+str(N_halos)+' most massive halos with FOF mass in DEUS-FUR, L_box=648Mpc/h, '+cosmo+'cdmw7, z = '+redshift+' \n')
    file_save.write('# halo identity, halo N_part, x (Mpc/h), y, z, cube numero, halo numero  \n')
    np.savetxt(file_save,prop_pre_sel)
    #prop_saved.to_csv(file_save,index=False)
    file_save.close()
    return()

def sel_most_massive_Mvir(cosmo='lcdm',redshift_str='1',N_sel=256) :
    path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+redshift_str+'/'+cosmo+'/most_massive/halo_properties/'
    names_all = np.array(pd.read_csv(path+'names_all.dat'))
    prop_all = np.array(pd.read_csv(path+'properties_all.dat'))
    prop_num_all = np.array(pd.read_csv(path+'properties_numbers_all.dat'))
    R_and_M_all = np.array(pd.read_csv(path+'R_and_M_all.dat'))
    Nvir_all = prop_num_all[:,1]
    Nfof_all = R_and_M_all[:,-1]
    ind_sel = np.argsort(Nvir_all)[-N_sel:]
    Nfof_sel = Nfof_all[ind_sel]
    Nfof_min_sel = np.min(Nfof_sel)
    Nfof_all_sort = np.sort(Nfof_all)
    ind_fof_min = np.where(Nfof_all_sort == Nfof_min_sel)[0]
    if ind_fof_min[0] < 9000 :
        print('Be carefull !!!')
        print(ind_fof_min)
    print(ind_fof_min)
    names_sel = names_all[ind_sel]
    for_save = str(N_sel)+' halos with the larger Mvir Brian and Norman from beyond FOF data, names, '+cosmo+', z ~ '+redshift_str
    names = pd.DataFrame(names_sel,columns=[for_save])
    names.to_csv(path+'names_sel.dat',index=False)
    R_and_M_sel = R_and_M_all[ind_sel]
    R_and_M_sel  = pd.DataFrame(R_and_M_sel,columns=['Rvir (Mpc/h)','Nvir','R200m','N200m','R200c','N200c',
                                                     'R500c','N500c','R1000c','N1000c','R2000c','N2000c','NFOF'])
    R_and_M_sel .to_csv(path+'R_and_M_sel.dat',index=False)
    prop_num_sel = prop_num_all[ind_sel]
    prop_num_sel = pd.DataFrame(prop_num_sel,columns=['Rvir (Mpc/h)','Nvir',
                                                      'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                                      'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                                      'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                                      'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                                      'cdm_x (Mpc/h)','cdm_y','cdm_z'])
    prop_num_sel.to_csv(path+'properties_numbers_sel.dat',index=False)
    prop_sel = prop_all[ind_sel]
    prop_sel = pd.DataFrame(prop_sel,columns=['halo name','Rvir (Mpc/h)','Nvir',
                                              'cdm_fof_x (Mpc/h)','cdm_fof_y','cdm_fof_z',
                                              'cdm_pot_x (Mpc/h)','cdm_pot_y','cdm_pot_z',
                                              'cdm_dens_x (Mpc/h)','cdm_dens_y','cdm_dens_z',
                                              'cdm_shrink_x (Mpc/h)','cdm_shrink_y','cdm_shrink_z',
                                              'cdm_x (Mpc/h)','cdm_y','cdm_z'])
    prop_sel.to_csv(path+'properties_sel.dat',index=False)
    return()

#sel_random_halos(mass_str='13',cosmo='wcdm',N_sel=256)

#concatenate_data(FOF='most_massive',redshift_str='2',cosmo='wcdm')

#sel_most_massive_Mvir(cosmo='wcdm',redshift_str='2',N_sel=256)

###########################################
# N_halos for z=0 for LCDM :  3045305
# N_halos for z=0 for RPCDM : 3066884
# N_halos for z=0 for wCDM :  3015407
###########################################
# N_halos for z=0.5 for LCDM :  3046603
# N_halos for z=0.5 for RPCDM : 2874911
# N_halos for z=0.5 for wCDM :  3079705
###########################################
# N_halos for z=1 for LCDM :  2888709
# N_halos for z=1 for RPCDM : 2518366
# N_halos for z=1 for wCDM :  2980683
###########################################
# N_halos for z=1.5 for LCDM : 2557330
# N_halos for z=1.5 for RPCDM : 2035971
# N_halos for z=1.5 for wCDM :  2684633
###########################################
# N_halos for z=2 for LCDM :  2108214
# N_halos for z=2 for RPCDM : 1511810
# N_halos for z=2 for wCDM :  2242684
cosmo = 'rp'
redshift = '0_5'
#path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+redshift+'/'+cosmo+'cdm/most_massive/halo_properties/'
path = '../../../../DEUSS_648Mpc_2048_particles/mass_bin_data/z_'+redshift+'/'+cosmo+'cdm/halos_position/halo_properties/'
N_halos = 2874911
#concatenate_halos(path,N_halos)
cosmo = 'rp'
redshift = '0_5'
N_halos = 10000
#sel_most_massive(cosmo,redshift,N_halos)


'''path = '/home/guillaume/Bureau/Thèse/DEUSS_648Mpc_2048_particles/mass_bin_data/mass_10_14.5_Msun_FOF/halo_properties_csv/'
prop = np.array(pd.read_csv(path+'properties_numbers_all.dat'))
#print(prop)

shift_approx = np.sqrt( (prop[:,3] - prop[:,6])**2 + (prop[:,4] - prop[:,7])**2 + (prop[:,5] - prop[:,8])**2 )
shift_approx /= prop[:,2]

dis_pot_dens = np.sqrt( (prop[:,9] - prop[:,6])**2 + (prop[:,10] - prop[:,7])**2 + (prop[:,11] - prop[:,8])**2 )

h_sml = 5 * 10**(-3)
res = 3 * h_sml
ind_unrelaxed = np.where(shift_approx > 0.07)[0]
ind_relaxed = np.where(shift_approx < 0.07)[0]
print(len(ind_unrelaxed))
N_halos = len(prop)
print(N_halos)
ind_pb = np.where(dis_pot_dens[ind_relaxed] > h_sml*2)[0]
print(len(ind_pb))
#for i in range(N_halos) :'''
    
