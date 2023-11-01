#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:30:56 2020

@author: guillaume
"""

##############"
### I want to compute the Inertia tensor, 
#############"

import numpy as np
import unsiotools.simulations.cfalcon as falcon
import time
import sys



# h=0.72 in the DEUSS simulations

Omega_matter_0 = 0.2573 # ratio of matter in the Universe at z=0, Omega_m = 0.257299989461899
z = -0.0003 # redshift  roughly 0 in DEUSS, for L = 648 Mpc/h, z = -0.00032383419689197
L = 648/(1+z) # size of thhe box in Mpc/h
mass = 2.2617 * 10**9 # mass of one particle in Msun/h, 2261715199.1513
H0 = 100 # Hubble constant in h km s**-1 Mpc**-1
G = 4.301 * 10**(-9) # Newton constant in Mpc km**2 s**-2 Msun**-1

Ez = np.sqrt(((1-Omega_matter_0) + Omega_matter_0 * (1 + z)**3)) # formula of the Hubble parameter as a function of z

Hz = H0 * Ez # formula of the Hubble expansion rate as a function of z in unit of h km s**-1 Mpc-1

Omega_matter_z = Omega_matter_0 * ((1 + z)**3) * ((1/Ez)**2) # ratio of matter in the Universe as a function of redshift

rho_crit = 3*Hz*Hz/(8*np.pi*G) # critical density at redshift z in Msun h**2 Mpc**-3

delta_vir_Brian_Norman = (18 * (np.pi)**2) + (82 * (Omega_matter_z - 1)) - (39 * (Omega_matter_z - 1)**2) # delta vir of Brian and Norman 1997 
rho_vir_Brian_Norman = delta_vir_Brian_Norman * Omega_matter_z * rho_crit # rho vir of Brian and Norman




    
    
def triaxiality_usual(x,y,z):
    
    ### technique of https://arxiv.org/abs/astro-ph/0508497
    N_particles = x.size
    
    factor = N_particles*mass
    #factor = 1
    
    I_11,I_22,I_33 = mass*np.sum(x**2)/factor,mass*np.sum(y**2)/factor,mass*np.sum(z**2)/factor

    I_12 = I_21 = mass*np.sum(x*y)/factor
    
    I_13 = I_31 = mass*np.sum(x*z)/factor
    
    I_23 = I_32 = mass*np.sum(y*z)/factor
    
    Inertia_tensor = np.matrix([[I_11,I_12,I_13],[I_21,I_22,I_23],[I_31,I_32,I_33]])
    
    Inertia_eigen_values,Passage = np.linalg.eig(Inertia_tensor)
    
    a,c = np.sqrt(max(Inertia_eigen_values)), np.sqrt(min(Inertia_eigen_values))
    
    for i in range(3):
        
        if a != np.sqrt(Inertia_eigen_values[i]) and c != np.sqrt(Inertia_eigen_values[i]) :
            
            b = np.sqrt(Inertia_eigen_values[i])
            
    Sphericity = c/a
    Elongation = b/a
    Triaxiality = (a**2 - b**2)/(a**2 - c**2)
    
    return(a,b,c,Sphericity,Elongation,Triaxiality,Passage,Inertia_tensor)
    
    
    
def triaxiality_with_r(x,y,z,r,N_particles):
    
    factor = N_particles*mass
    #factor = mass
    
    # mass tensor
    I_11,I_22,I_33 = mass*np.sum((x/r)**2)/factor,mass*np.sum((y/r)**2)/factor,mass*np.sum((z/r)**2)/factor
    I_12 = I_21 = mass*np.sum((x*y/r**2))/factor
    I_13 = I_31 = mass*np.sum((x*z/r**2))/factor
    I_23 = I_32 = mass*np.sum((y*z/r**2))/factor
    
    # diagonaliszation
    Inertia_tensor = np.matrix([[I_11,I_12,I_13],[I_21,I_22,I_23],[I_31,I_32,I_33]])
    Inertia_eigen_values,Passage = np.linalg.eig(Inertia_tensor) # it contains the eigenvalues and the passage matrix
    
    #print(np.dot(np.linalg.inv(Passage),np.dot(Inertia_tensor,Passage)))
    
    # length of the axis
    a,c = np.sqrt(max(Inertia_eigen_values)), np.sqrt(min(Inertia_eigen_values))
    for i in range(3):
        if a != np.sqrt(Inertia_eigen_values[i]) and c != np.sqrt(Inertia_eigen_values[i]) :
            b = np.sqrt(Inertia_eigen_values[i])
            
           
    # shape
    Sphericity,Elongation,Triaxiality = c/a,b/a,(a**2 - b**2)/(a**2 - c**2)
    
    return(a,b,c,Sphericity,Elongation,Triaxiality,Passage,Inertia_tensor)
            


def highest_density_JC_dehnen(N,x,y,z,N_nearest_neighbours,N_ferrer):
    
    mass_array = np.float32(np.ones(N) * mass)
    
    pos = np.append(np.append(x,y),z)
    
    pos = pos.reshape(3,N).transpose()
    
    pos = pos.reshape(1,3*N)[0]
    
    pos = np.float32(pos)
    
    cf = falcon.CFalcon()

    ok,rho,hsml = cf.getDensity(pos,mass_array,N_nearest_neighbours,N_ferrer)
    
    return(rho)
    
def periodicity(data):
    
    x = data[0] # position in box unit (between 0 and 1), it is a numpy array
    y = data[1]
    z = data[2]
    
    #print(x.size)
    
    # if there are particles very far away from each other, I put their numero here
    # I take the very first particle as refernce, it does not change anything to choose any particle
    bad_x = np.empty([0],dtype=int)
    bad_y = np.empty([0],dtype=int)
    bad_z = np.empty([0],dtype=int)

    y_new = y # I will transform x_new if there are periodicity problems
    x_new = x
    z_new = z

    bad_periodicity = 1 # If there is NO periodicity problem, it goes to zero, otherwise it stays to one

    for p in range(1,x.size) :
        
        if np.abs(x[p] - x[0]) > 0.5 :
            
            bad_x = np.append(bad_x,[p])
            
        elif np.abs(y[p] - y[0]) > 0.5 :
                  
            bad_y = np.append(bad_y,[p])
        
        elif np.abs(z[p] - z[0]) > 0.5 :
            
            bad_z = np.append(bad_z,[p])
            
    
    # if there is NO problem  
    if bad_x.size == 0 and bad_y.size == 0 and bad_z.size == 0 :
    
        bad_periodicity = 0 # NO periodicity problem
        
    # otherwise
    else :
        
        # if there is a problem on x
        if bad_x.size != 0 :
        
            for i in range(bad_x.size) :
            
                if x[bad_x[i]] > 0.9 :
                
                    x_new[bad_x[i]] = x[bad_x[i]] - 1
                
                else :
                
                    x_new[bad_x[i]] = 1 + x[bad_x[i]]
        
        # if there is a problem on y
        elif bad_y.size != 0 :
        
            for i in range(bad_y.size) :
            
                if y[bad_y[i]] > 0.9 :
                
                    y_new[bad_y[i]] = y[bad_y[i]] - 1
                
                else :
                
                    y_new[bad_y[i]] = 1 + y[bad_y[i]]
        
        # if there is a problem on z
        else :
        
            for i in range(bad_z.size) :
            
                if z[bad_z[i]] > 0.9 :
                
                    z_new[bad_z[i]] = z[bad_z[i]] - 1
                
                else :
                
                    z_new[bad_z[i]] = 1 + z[bad_z[i]]  
                    
    # x_new are the positions, numpy arrays
    return(x_new,y_new,z_new,bad_periodicity)
    
    
    
path = '../mass_10_14_Msun'

#path_properties = './output/'

#name = 'fof_boxlen648_n2048_lcdmw7_strct_00000_02114.dat'
#halo_numero = 0

#data = np.loadtxt(''+path+'/'+name+'')

#N_haloes = 2862






def compute_ellipse_isodensity(N_haloes,N_neighbours):
    # I compute the shape following: https://arxiv.org/abs/astro-ph/0202064
    # I select particles with density less than rho_vir and I do not throw away the substructures
    
    halo_axis = np.array(())
    halo_axis_full = np.array(())

    for h in range(N_haloes):
    
        #print(h)
    
        #I put the properties of my halo in my_halo
        my_halo = data_properties[h] 
        name = my_halo[0] # name of the halo
        cdm = [np.float(my_halo[-10]),np.float(my_halo[-9]),np.float(my_halo[-8])] # highest density point of my halo, that I use as center
        
        # data contains the positions, velocities and identities of all the particles given by the FOF
        data = np.loadtxt(''+path+'/'+name+'')
        N_particles = np.int(data.size/7) # number of particless in the halo
        positions = data[:3*N_particles] # positions of all particles
        positions = positions.reshape(N_particles,3).transpose() # I reshape the position
            
        # I controll the periodicity, the halo could be at a border of the simulation box and have particles on  both sides
        x_new,y_new,z_new,bad_periodicity = periodicity(positions) 
            
        ########### I create isodensity
        
        # I put the positions in Mpc/h in the frame of the halo    
        x,y,z = x_new*L-cdm[0],y_new*L-cdm[1],z_new*L-cdm[2]
        
        # I compute the density associated to each particle by using the Dehnen function, with N_neighbours nearest neighbours
        rho = highest_density_JC_dehnen(N_particles,x,y,z,N_neighbours,0)
        
        #I sort the density
        ind = np.argsort(rho)
        rho_sort = rho[ind]
        
        # I take only the particles that have a density in the range factor_dens_grt*rho_vir > rho_i > factor_dens_lss*rho_vir
        factor_dens_lss = 1
        high_density_indice = np.where(rho_sort > factor_dens_lss*rho_vir_Brian_Norman)
        ind_used = ind[high_density_indice]
        # I put in x_used the position of particles which a density greater than rho_vir
        x_used,y_used,z_used = x[ind_used],y[ind_used],z[ind_used]
                
        # I compute the inertia tensor, the axis length and the sphericity, elongation and triaxiality and the Passage matrix
        usual = triaxiality_usual(x_used,y_used,z_used)
        #print(usual)
        #print(usual[-4],usual[-3],usual[-2])
            
        halo_axis = np.append(halo_axis,usual)
        halo_axis_full = np.append(halo_axis_full,np.append(np.array((name)),usual))
    
    halo_axis = halo_axis.reshape(N_haloes,6+1)
    halo_axis_full = halo_axis_full.reshape(N_haloes,7+1)
    
    #print(usual[-4],usual[-3],usual[-2])
    #print(halo_axis)

    #np.savetxt('halo_axis.dat',halo_axis)
    np.savetxt('halo_axis_full.dat',halo_axis_full,fmt='%s')
    

    return(halo_axis,halo_axis_full)
    
    
def compute_ellipse_simple(N_haloes_begin):
    # I compute the shape following: https://arxiv.org/pdf/astro-ph/0408163.pdf
    
    path_properties = './output/'
    
    data_properties = np.loadtxt(''+path_properties+'data_properties_'+str(N_haloes_begin + 1)+'.dat',dtype={ # I specify the type of data 
                      'names': ('halo identity','N','N_Brian_Norman_first','N_Neto_first','N_200_first','cdm[0]','cdm[1]','cdm[2]','r_first','r_last,r200','r_Brian_Norman','r_Neto',
                                'N_Brian_Norman_second','N_Neto_second','N_200_second','CDM[0]','CDM[1]','CDM[2]','R_first','R_last','R200','R_Brian_Norman','R_Neto',
                                'N_Brian_Norman_third','N_Neto_third','N_200_third','cdm_third[0]','cdm_third[1]','cdm_third[2]','r_first_third','r_last_third','r200_third','r_Brian_Norman_third','r_Neto_third','indice_Power','Rpower','bidule?????'),
                      'formats': ('U50',np.float, np.float, np.float, np.float,np.float,np.float,np.float, np.float, np.float, np.float,np.float,np.float,
                                  np.float, np.float, np.float, np.float,np.float,np.float,np.float, np.float, np.float, np.float,np.float,
                                  np.float,np.float, np.float, np.float, np.float,np.float,np.float,np.float, np.float, np.float, np.float,np.float,np.float,np.float)},unpack=True)


    data_properties = np.array(data_properties).transpose()
    
    path = '../mass_10_14_Msun'
    N_haloes = len(data_properties)
        
    
    #halo_axis = np.array(())
    halo_axis_full = np.array(())
    
    q = s = np.array((1))
    #print(q)

    for h in range(N_haloes):
    
        #print(h)
    
        #I put the properties of my halo in my_halo
        my_halo = data_properties[h] 
        name = my_halo[0] # name of the halo
        cdm = [np.float(my_halo[-10]),np.float(my_halo[-9]),np.float(my_halo[-8])] # center of mass weighted by the density associated to each particles
        Rvir = np.float(my_halo[-4]) # Virial radius (Brian and Norman) in Mpc/h
        #print(Rvir)
        
        # data contains the positions, velocities and identities of all the particles given by the FOF
        data = np.loadtxt(''+path+'/'+name+'')
        N_particles = np.int(data.size/7) # number of particless in the halo
        positions = data[:3*N_particles] # positions of all particles
        positions = positions.reshape(N_particles,3).transpose() # I reshape the position
        
        # I controll the periodicity, the halo could be at a border of the simulation box and have particles on  both sides
        x_new,y_new,z_new,bad_periodicity = periodicity(positions) 
            
        # I put the positions in Mpc/h in the frame of the halo    
        x,y,z = x_new*L-cdm[0],y_new*L-cdm[1],z_new*L-cdm[2]
        
        # I compute the radius (distance to the center) of each particle
        r = np.sqrt(x**2 + (y/q)**2 + (z/s)**2)
        
        ind = np.where(r<Rvir)
        x,y,z,r = x[ind],y[ind],z[ind],r[ind]
        N_particles = x.size
        #print(N_particles)
        
        # I compute the shape
        a,b,c,Sphericity,Elongation,Triaxiality,Passage,Inertia_tensor = triaxiality_with_r(x,y,z,r,N_particles)
        
        # I rescale following: https://arxiv.org/pdf/astro-ph/0408163.pdf
        Sphericity,Elongation = Sphericity**(np.sqrt(3)),Elongation**(np.sqrt(3))
        Triaxiality = (1-Elongation**2)/(1-Sphericity**2)
        
        # I put the results here
        simple = np.array((a,b,c,Sphericity,Elongation,Triaxiality,Passage))
        #halo_axis = np.append(halo_axis,simple)
        halo_axis_full = np.append(halo_axis_full,np.append(np.array((name)),simple))
        
        #print(a,b,c,Sphericity,Elongation,Triaxiality)
    
    #halo_axis = halo_axis.reshape(N_haloes,6)
    halo_axis_full = halo_axis_full.reshape(N_haloes,8)
    
    #print(halo_axis_full)

    #np.savetxt('halo_axis.dat',halo_axis)
    np.savetxt('./output/shape_simple_properties_'+str(N_haloes_begin)+'.dat',halo_axis_full,fmt='%s')

    return(halo_axis_full)
    
    
    
def compute_ellipse_convergence(halo_numero_begin,Initial_number_iteration,error_initial,time_duration_limit):
    # I compute the shape following: https://arxiv.org/pdf/astro-ph/0508497.pdf: 
    # Allgood et al 2005, The Shape of Dark Matter Halos: Dependence on Mass,Redshift, Radius, and Formation
    
    # halo_numero_begin: numero of the first halo from which I want to compute the shape
    # Initial_number_iteration: number of iteration of my initialization, 
    # it is also the number of values of sphericity and elongation that I compare to see if it converged
    # error_initial: it is the initial error that I accept for the sphericity and elongation in percentage 
    # it can be more at the end if it does not converge for this error, I will increase the error 
    # time_duration_limit: it is the time duration (in second) where it looks for convergence 
    # I think that if it does not converge after a few second, it won't converge at this error even after a very long time but I am not sure
    
    # I load properties on all my haloes
    path_properties = './output/'
    data_properties = np.loadtxt(''+path_properties+'data_properties_'+str(halo_numero_begin + 1)+'.dat',dtype={ # I specify the type of data 
                      'names': ('halo identity','N','N_Brian_Norman_first','N_Neto_first','N_200_first','cdm[0]','cdm[1]','cdm[2]','r_first','r_last,r200','r_Brian_Norman','r_Neto',
                                'N_Brian_Norman_second','N_Neto_second','N_200_second','CDM[0]','CDM[1]','CDM[2]','R_first','R_last','R200','R_Brian_Norman','R_Neto',
                                'N_Brian_Norman_third','N_Neto_third','N_200_third','cdm_third[0]','cdm_third[1]','cdm_third[2]','r_first_third','r_last_third','r200_third','r_Brian_Norman_third','r_Neto_third','indice_Power','Rpower','bidule?????'),
                      'formats': ('U50',np.float, np.float, np.float, np.float,np.float,np.float,np.float, np.float, np.float, np.float,np.float,np.float,
                                  np.float, np.float, np.float, np.float,np.float,np.float,np.float, np.float, np.float, np.float,np.float,
                                  np.float,np.float, np.float, np.float, np.float,np.float,np.float,np.float, np.float, np.float, np.float,np.float,np.float,np.float)},unpack=True)

    # I reshape it
    data_properties = np.array(data_properties).transpose()
    # the path to the haloes
    path_to_data = '../mass_10_14_Msun'
    # number of haloes
    N_haloes = len(data_properties)
        
    halo_axis_full = np.array(()) # I will put all the results that I want to keep here
    for h in range(N_haloes):
        
        print(h) # just to know how many haloes have been computed
        
        ##### loading of my halo
        # I put the properties of the halo h in my_halo
        my_halo = data_properties[h] 
        name = my_halo[0] # name of the halo
        cdm = [np.float(my_halo[-10]),np.float(my_halo[-9]),np.float(my_halo[-8])] # center of mass of my halo weighted by the density associated to each particles in Mpc/h
        Rvir = np.float(my_halo[-4]) # Virial radius (Brian and Norman) in Mpc/h
        # data contains the positions, velocities and identities of all the particles given by the FOF
        data = np.loadtxt(''+path_to_data+'/'+name+'')
        N_particles = np.int(data.size/7) # number of particless in the halo
        positions = data[:3*N_particles] # positions of all particles in boxlength unit
        positions = positions.reshape(N_particles,3).transpose() # I reshape the position
        # I controll the periodicity, the halo could be at a border of the simulation box and have particles on both sides
        x_new,y_new,z_new,bad_periodicity = periodicity(positions) 
        # I put the positions in unit of Rvir in the frame of the halo, L is there to put the position in Mpc/h   
        x,y,z = (x_new*L-cdm[0])/Rvir,(y_new*L-cdm[1])/Rvir,(z_new*L-cdm[2])/Rvir
        
        ###### Initialization
        a = b = c = 1 # axis length in unit of Rvir of the ellipse, it will change at each iteration
        Sphericity, Elongation = np.array((c/a)),np.array((b/a)) # I will put all the values of sphericty and elongation here, for each iteration
        Passage = np.identity(3) # passage matrix between two frame, it will change at each iteration
        error = error_initial # at the begining, I try to get convergence with error_initial, if it does not work I will try with a higher error
        # initial inverse of passage matrix, it will contain the inverse global passage matrix between the original frame and the final frame
        Passage_general_inverse = np.linalg.inv(Passage) 
        
        # I initialize a first short loop of Initial_number_iteration
        for i in range(Initial_number_iteration):
            # I put the particles position in a matrix 3*N_particles
            position = np.matrix(np.array((x,y,z)))
            # I multiply the position by the passage matrix, to go in the natural frame of the ellipsoid
            position_new = np.array((np.dot(np.linalg.inv(Passage),position)))
            x,y,z = position_new[0],position_new[1],position_new[2]
                        
            # I compute a radius for each particle, weighted by the ellipsoid axis
            r_ellipse = np.sqrt(x**2 + (y/(b/a))**2 + (z/(c/a))**2)
            # I select only particles inside the ellipsoid, all in unit of Rvir
            ind_ellipse = np.where(r_ellipse<1)
            x,y,z,r_ellipse = x[ind_ellipse],y[ind_ellipse],z[ind_ellipse],r_ellipse[ind_ellipse]
            N_particles_ellipse = x.size
            
            # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
            a,b,c,s,q,Triaxiality,Passage,Inertia_tensor = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
            
            # I add a new value of sphericity and Elongation
            Sphericity, Elongation = np.append(Sphericity,c/a), np.append(Elongation,b/a)
            # I put the global inverse of the passage matrix, to go from the initial frame to the final one here
            Passage_general_inverse = np.dot(np.linalg.inv(Passage),Passage_general_inverse)
            
            # If the particles number becomes significantly lower, then I stop
            if N_particles_ellipse < N_particles/2 :
                
                error=0 # to say that there was a problem
                
                simple = np.array((a,b,c,c/a,b/a,Triaxiality,np.linalg.inv(Passage_general_inverse),error,Initial_number_iteration))
                
                break # stop the loop
            
        time_begin = time.time() # time where I begin the while loop
        # I do a while loop, and try to find a convergence for Sphericity and Elongation values 
        # I want that the Initial_number_iteration values of Sphericity and Elongation to be very close from each other
        while np.abs(sum(Sphericity[-Initial_number_iteration:])/(Initial_number_iteration * Sphericity[-1]) - 1) > error or np.abs(sum(Elongation[-Initial_number_iteration:])/(Initial_number_iteration * Elongation[-1]) - 1) > error :
            # I put the particles position in a matrix 3*N_particles
            position = np.matrix(np.array((x,y,z)))
            # I multiply the position by the inverse passage matrix, to go in the natural frame of the ellipsoid
            position_new = np.array((np.dot(np.linalg.inv(Passage),position)))
            x,y,z = position_new[0],position_new[1],position_new[2]
                        
            # I compute a radius for each particle, weighted by the ellipsoid axis
            r_ellipse = np.sqrt(x**2 + (y/(b/a))**2 + (z/(c/a))**2)
            # I select particles inside the ellipsoid, all in unit of Rvir
            ind_ellipse = np.where(r_ellipse<1)
            x,y,z,r_ellipse = x[ind_ellipse],y[ind_ellipse],z[ind_ellipse],r_ellipse[ind_ellipse]
            N_particles_ellipse = x.size
                        
            # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
            a,b,c,s,q,Triaxiality,Passage,Inertia_tensor = triaxiality_with_r(x,y,z,r_ellipse,N_particles_ellipse)
            
            # I add a new value 
            Sphericity, Elongation = np.append(Sphericity,c/a), np.append(Elongation,b/a)
            # I put the global inverse of the passage matrix, to go from the initial frame to the final one here
            Passage_general_inverse = np.dot(np.linalg.inv(Passage),Passage_general_inverse)
            
            time_duration = time.time() - time_begin # time duration from the begining of the while loop
            # If the time to get convergence is too important, it means that it won't converge and so I increase the error and retry with this new error
            if time_duration > time_duration_limit:
                error = error*1.1 # new error
                time_begin = time.time() # new begining time of the while loop
                
            # If the particles number becomes significantly lower, then I stop
            if N_particles_ellipse < N_particles/2 :
                error=0 # to say that there was a problem
                
                simple = np.array((a,b,c,c/a,b/a,Triaxiality,np.linalg.inv(Passage_general_inverse),error,Initial_number_iteration))
                
                break

        
        
        # I put the results here
        simple = np.array((a,b,c,c/a,b/a,Triaxiality,np.linalg.inv(Passage_general_inverse),error,Initial_number_iteration))
        
        halo_axis_full = np.append(halo_axis_full,np.append(np.array((name)),simple))
        
    
    # I reshape the final result and save it
    halo_axis_full = halo_axis_full.reshape(N_haloes,10)
    np.savetxt('./output/shape_CV_properties_'+str(halo_numero_begin)+'.dat',halo_axis_full,fmt='%s')

    return(halo_axis_full)
    
#n_parallel = np.int(sys.argv[1]) # I import a number from a terminal command line
n_parallel = 0   

compute_ellipse_convergence(n_parallel,100,0.001,10)

compute_ellipse_simple(n_parallel)






