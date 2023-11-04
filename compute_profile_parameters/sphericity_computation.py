#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 14:36 2023

@author: guillaume
"""

"""
This script computes the sphericity parameters of a halo
"""
import numpy as np
from typing import Dict

from semi_analytical_halos.generate_smooth_halo import Smooth_halo

class Sphericity():

    def compute_SQT_once(self, pos: np.ndarray[float],
                         r: np.ndarray[float], N_particles: int,) -> Dict[str, float]:
        # see Zemp et al: https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
        # shape tensor, see equation 8 in Zemp et al, 
        # where we assume that all our particles have the same mass
        I_11, I_22, I_33 = np.sum((pos[:,0]/r)**2), np.sum((pos[:,1]/r)**2), np.sum((pos[:,2]/r)**2)
        I_12 = I_21 = np.sum(pos[:,0]*pos[:,1]/r**2)
        I_13 = I_31 = np.sum(pos[:,0]*pos[:,2]/r**2)
        I_23 = I_32 = np.sum(pos[:,1]*pos[:,2]/r**2)
        shape_tensor = np.matrix([[I_11, I_12, I_13], [I_21, I_22, I_23], [I_31, I_32, I_33]])/N_particles
        # diagonaliszation
        eigen_values, evecs = np.linalg.eig(shape_tensor) # it contains the eigenvalues and the passage matrix
        if eigen_values.any() < 0 :
            print('aieeeeeee!!!!')
        # obtain the a, b, c, S, Q, T parameters from eigenvalues and the Passage matrix
        eigen_values *= 3
        #eigen_values *= 5
        eigen_values = np.sqrt(eigen_values)
        order = np.argsort(eigen_values) # order the eigen values, see Zemp et al: https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
        Passage = np.identity(3)
        Passage[:,0],Passage[:,1],Passage[:,2] = evecs[order[2]], evecs[order[1]], evecs[order[0]]
        a, b, c = eigen_values[order[2]], eigen_values[order[1]], eigen_values[order[0]]
        # shape
        Sphericity, Elongation, Triaxiality = c/a, b/a, (a**2 - b**2)/(a**2 - c**2)
        dic = {"a": a, "b": b, "c": c,
            "Sphericity": Sphericity, "Elongation": Elongation, "Triaxiality": Triaxiality,
            "Passage": Passage}
        return dic
    
    def initialization(self, pos: np.ndarray[float], it_min=10,) -> Dict[str, np.ndarray[float]]:
        N_particles = len(pos)
        #a, b, c = 1, 1, 1 # axis length in unit of the largest axis of the ellipsoid, it will change at each iteration
        a_arr, b_arr, c_arr = np.ones((it_min)), np.ones((it_min)), np.ones((it_min)), np.ones((it_min))
        Sphericity, Elongation, Triaxiality = np.ones((it_min)), np.ones((it_min)), np.ones((it_min)) # all the values of sphericty and elongation will be here, for each iteration
        Passage, P_inv = np.ones((it_min, 3, 3)), np.ones((it_min, 3, 3))
        #Passage = np.identity(3) # passage matrix between two frame, it will change at each iteration
        #P_inv = np.linalg.inv(Passage) # inverse passage matrice between 2 frames, it will cahnge at each iteration
        Passage_gen, P_inv_gen = np.identity(3), np.identity(3)
        #P_inv_gen = np.identity(3) # general inverse matrice: it corresponds to the change between the first and the last frame
        #r_ell = np.sqrt(pos[:,0]**2 + (pos[:,1]/Elongation[-1])**2 + (pos[:,2]/Sphericity[-1])**2 ) # ellipsoidal radius
        for it in range(1, it_min):
            r_ell = np.sqrt(pos[:,0]**2 + (pos[:,1]/Elongation[-1])**2 + (pos[:,2]/Sphericity[-1])**2 )
            # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
            res = self.compute_SQT_once(pos, r_ell, N_particles)
            # I add a new value of sphericity and Elongation
            Sphericity[it], Elongation[it] = res["Sphericity"], res["Elongation"]
            a_arr[it], b_arr[it], c_arr[it], Triaxiality[it] = res["a"], res["b"], res["c"], res["Triaxiality"]
            Passage[it] = res["Passage"]
            P_inv[it] = np.linalg.inv(Passage[it]) # inverse passage between 2 consecutive frames
            # I multiply the position by the inverse passage matrix, 
            # in order to go in the natural frame of the ellipsoid
            pos = np.array((np.dot(P_inv[it], pos)))
            #x, y, z = pos[0], pos[1], pos[2]
            # I compute a radius for each particle, weighted by the ellipsoid axis
            #r_ell = np.sqrt(x**2 + (y/Elongation[-1])**2 + (z/Sphericity[-1])**2) 
            # global inverse passage matrice
            Passage_gen = np.dot(Passage[it], Passage_gen) 
            P_inv_gen = np.dot(P_inv[it], P_inv_gen) 
        # I put the particles position in a matrix 3*N_particles
        dic = {"a": a_arr, "b": b_arr, "c": c_arr,
               "Sphericity": Sphericity, "Elongation": Elongation, "Triaxiality": Triaxiality,
               "Passage": Passage, "P_inv": P_inv, "Passage_gen": Passage_gen, "P_inv_gen": P_inv_gen,
               "r_ell": r_ell, "N_particles": N_particles, "pos": pos}
        return dic
    
    def update_once(self, pos, r_ell, a, a_unit=1):
        ind_ell = np.where(r_ell < a/a_unit )[0]
        x, y, z, r_ell = pos[ind_ell, 0], pos[ind_ell, 1], pos[ind_ell, 2], r_ell[ind_ell]
        N_particles_ellipse = x.size
        # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
        a, b, c, s, q, t, Passage = self.compute_SQT_once(x, y, z, r_ell, N_particles_ellipse)
        # I add a new value of sphericity and Elongation
        #Sphericity, Elongation = np.append(Sphericity, s), np.append(Elongation, q)
        P_inv = np.linalg.inv(Passage) # inverse passage between 2 consecutive frames
        # global inverse passage matrice
        pos = np.array((np.dot(P_inv, np.matrix(pos.T)))).T
        #x, y, z = pos[0], pos[1], pos[2]
        # I compute a radius for each particle, weighted by the ellipsoid axis
        r_ell = np.sqrt(pos[:,0]**2 + (pos[:,1]/q)**2 + (pos[:,2]/s)**2 ) 
        Passage_general_inverse = np.dot(P_inv, Passage_general_inverse) 
        dic = {"a": a, "b": b, "c": c, "s": s, "q": q, "t": t,
               "Passage_general_inverse": Passage_general_inverse,
               "pos": pos, "r_ell": r_ell}
        return dic
    
    def run_many(self, pos, it_min=10, it_loop=10, error=0.01):
        res = self.initialization(pos, it_min=it_min)
        Sphericity, Elongation = res["Sphericity"], res["Elongation"]
        Triaxiality = res["Triaxiality"]
        pos = res["pos"]
        r_ell = res["r_ell"]
        a_arr = res["a_arr"]
        it = 0
        err_Sphericity = np.std(Sphericity[-it_min:])/np.mean(Sphericity[-it_min:])
        err_Elongation = np.std(Elongation[-it_min:])/np.mean(Elongation[-it_min:])
        while (it < it_min) or ((err_Sphericity > error) or (err_Elongation > error)):
            ind_ell = np.where(r_ell < a_arr[-1])[0]
            #x, y, z, r_ell = pos[ind_ell, 0], pos[ind_ell, 1], pos[ind_ell, 2], r_ell[ind_ell]
            N_particles_ellipse = len(ind_ell)
            # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
            res_once = self.compute_SQT_once(pos[ind_ell], r_ell[ind_ell], N_particles_ellipse)
            Sphericity = np.append(Sphericity, res_once["Sphericity"])
            Elongation = np.append(Elongation, res_once["Elongation"])
            Triaxiality = np.append(Triaxiality, res_once["Triaxiality"])
            a_arr = np.append(a_arr, res_once["a"])
            b_arr = np.append(b_arr, res_once["b"])
            c_arr = np.append(c_arr, res_once["c"])
            Passage = np.append(Passage, res_once["Passage"])
            P_inv = np.append(P_inv, res_once["P_inv"])
            
            Passage[it] = res["Passage"]
            P_inv[it] = np.linalg.inv(Passage[it]) # inverse passage between 2 consecutive frames
            # I add a new value of sphericity and Elongation
            #Sphericity, Elongation = np.append(Sphericity, s), np.append(Elongation, q)
            P_inv = np.linalg.inv(Passage) # inverse passage between 2 consecutive frames
            # global inverse passage matrice
            pos = np.array((np.dot(P_inv, np.matrix(pos.T)))).T
            #x, y, z = pos[0], pos[1], pos[2]
            # I compute a radius for each particle, weighted by the ellipsoid axis
            r_ell = np.sqrt(pos[:,0]**2 + (pos[:,1]/q)**2 + (pos[:,2]/s)**2 ) 
            Passage_general_inverse = np.dot(P_inv, Passage_general_inverse) 
            #if it > 
            Sphericity
            it += 1
            err_Sphericity = np.std(Sphericity[-it_min:])/np.mean(Sphericity[-it_min:])
            err_Elongation = np.std(Sphericity[-it_min:])/np.mean(Sphericity[-it_min:])
            if it - it_min > it_loop:
                error *= 1.1
                
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


if __name__ == '__main__':
    halo = Smooth_halo()
    N_part_th = 10000
    N_bin_th = 20
    R_max_th = 1/0.6
    res_th = 0.01
    a_ax_th = 1
    b_ax_th = 0.8
    c_ax_th = 0.6
    S_th = c_ax_th/a_ax_th
    print("R_max_th =", R_max_th)
    print("S_th =", S_th)
    Q_th = b_ax_th/a_ax_th
    print("Q_th =", Q_th)
    if a_ax_th**2 - c_ax_th**2 != 0:
        T_th = (a_ax_th**2 - b_ax_th**2)/(a_ax_th**2 - c_ax_th**2)
        print("T_th =", T_th)
    else:
        print("T_th does not exist")

    my_halo = halo.smooth_halo_creation(N_part=N_part_th, N_bin=N_bin_th, R_max=R_max_th, res=res_th,
                                        a_ax=a_ax_th, b_ax=b_ax_th, c_ax=c_ax_th)
    data = my_halo["data"]
    halo.beauty_plot_colorbar(data, ax_1=0, ax_2=2, r_max=R_max_th)
    sphere = Sphericity()
    res = sphere.compute_SQT_once(pos=data, r=my_halo["r"], N_particles=my_halo["N_tot"])
    print(res)