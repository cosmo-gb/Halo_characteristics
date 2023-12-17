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

from semi_analytical_halos.generate_smooth_halo import Smooth_halo

class Sphericity():
    """
    This class allows to compute the shape of a halo
    """

    def __init__(self,):
        # compute the shape of an ellipsoid (ellipsoid) or an homeoid (homeoid)
        self.ellipsoidal_kind: str="ellipsoid"
        # number of loops for initialization and average of S and Q on iterations
        self.it_init: int=10
        # increase error threshold every it_loop
        self.it_loop: int=20 # can increase after some iterations
        self.it_loop_fixed: int=20 # fixed
        # error threshold initially
        self.error: float=0.001
        # fraction of particles that must remain, otherwise stop the computation
        self.N_part_frac: float=0.1

    def compute_SQT_once(self, pos: np.ndarray[float],
                         radius: np.ndarray[float],
                         N_particles: int,
                         ) -> dict[str, float]:
        """
        Computes the shape tensor of an object with 3D particle positions given by pos
        then computes the semi principal axis length, a > b > c
        then computes the sphericity, elongation and trixiality coefficient
        and finally computes the passage matrix made of the eigen vectors of the shape tensor
        # input:
        - pos: np.ndarray[float], particles positions of the object considered, shape=N_particles, 3
        - radius: np.ndarray[float], particles radius within the object, shape=N_particles
        note that 3 cases have been considered by Zemp et al: radius=1, r, r_ell
        in this script we are using radius = r_ell, but this function could be used in the 2 other cases
        - N_particles: int, number of particles in the object
        # output:
        - a > b > c: float, semi-principal axis length in object size unit
        - S, Q, T: float, sphericity, elongation and trixiality coefficient
        - Passage: np.ndarray[float], shape=3,3, passage matrix of the shape tensor
        """
        # see Zemp et al: https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
        # shape tensor, see equation 8 in Zemp et al, 
        # where we assume that all our particles have the same mass
        I_11, I_22, I_33 = np.sum((pos[:,0]/radius)**2), np.sum((pos[:,1]/radius)**2), np.sum((pos[:,2]/radius)**2)
        I_12 = I_21 = np.sum(pos[:,0]*pos[:,1]/radius**2)
        I_13 = I_31 = np.sum(pos[:,0]*pos[:,2]/radius**2)
        I_23 = I_32 = np.sum(pos[:,1]*pos[:,2]/radius**2)
        shape_tensor = np.matrix([[I_11, I_12, I_13], [I_21, I_22, I_23], [I_31, I_32, I_33]])/N_particles
        # diagonaliszation
        eigen_values, evecs = np.linalg.eig(shape_tensor) # it contains the eigenvalues and the passage matrix
        if eigen_values.any() < 0 :
            print("The eigenvalues should be positives")
        # obtain the a, b, c, S, Q, T parameters from eigenvalues and the Passage matrix
        # see equation 8 and below in Zemp et al (even if it seems any factor is ok)
        if self.ellipsoidal_kind == "ellipsoid": # case of a full ellipsoid, see Zemp et al.
            eigen_values *= 5
        elif self.ellipsoidal_kind == "shell": # case of an ellipsoidal shell, i.e. an homoeoid, see Zemp et al.
            eigen_values *= 3
        else:
            eigen_values = eigen_values
        eigen_values = np.sqrt(eigen_values)
        # order the eigen values, see Zemp et al: https://arxiv.org/pdf/1107.5582.pdf#cite.1996MNRAS.278..488Z
        order = np.argsort(eigen_values)
        Passage = np.identity(3)
        Passage[:,0], Passage[:,1], Passage[:,2] = evecs[order[2]], evecs[order[1]], evecs[order[0]]
        a, b, c = eigen_values[order[2]], eigen_values[order[1]], eigen_values[order[0]]
        # compute shape
        Sphericity, Elongation, Triaxiality = c/a, b/a, (a**2 - b**2)/(a**2 - c**2)
        dic = {"a": a, "b": b, "c": c,
               "Sphericity": Sphericity, "Elongation": Elongation, "Triaxiality": Triaxiality,
               "Passage": Passage}
        return dic
    
    def SQT_initialization(self, pos: np.ndarray[float],
                           N_particles: int,
                           ) -> dict[str, np.ndarray[float]]:
        """
        Initialize computation of the object shape, before to try to converge
        call self.it_init - 1 times the method compute_SQT_once
        returns all the shape parameters at each iteration and more
        # input:
        - pos: np.ndarray[float], shape=(N_part, 3), particle positions
        - N_particles: int, number of particles in the object
        # output:
        - a_arr, b_arr, c_arr: np.ndarray[float], shape=self.it_init, 
        semi principal axis length at each iteration
        - Sphericity, Elongation, Triaxiality: np.ndarray[float], shape=self.it_init,
        shape coefficients at each iteration
        - Passage, P_inv: np.ndarray[float], shape=(self.it_init, 3, 3)
        passage and inverse passage matrix at each iteration
        - Passage_gen, P_inv_gen: np.ndarray[float], shape=(3, 3)
        gloabl passage and inverse passage matrix (combined all iterations)
        - r_ell: np.ndarray[float], shape=N_particles, ellipsoidal radius of the last iteration
        - pos: np.ndarray[float], shape=(N_part, 3), particle positions at the last iteration
        """
        a_arr, b_arr, c_arr = np.ones((self.it_init)), np.ones((self.it_init)), np.ones((self.it_init))
        Sphericity, Elongation, Triaxiality = np.ones((self.it_init)), np.ones((self.it_init)), np.ones((self.it_init)) # all the values of sphericty and elongation will be here, for each iteration
        Passage, P_inv = np.ones((self.it_init, 3, 3)), np.ones((self.it_init, 3, 3))
        Passage_gen, P_inv_gen = np.identity(3), np.identity(3)
        for it in range(1, self.it_init):
            r_ell = np.sqrt(pos[:,0]**2 + (pos[:,1]/Elongation[it-1])**2 + (pos[:,2]/Sphericity[it-1])**2 )
            # I compute the shape, here a > b > c have values in unit of Rvir, a can be different from 1
            res = self.compute_SQT_once(pos, r_ell, N_particles)
            # I add a new value of sphericity and Elongation
            Sphericity[it], Elongation[it] = res["Sphericity"], res["Elongation"]
            a_arr[it], b_arr[it], c_arr[it], Triaxiality[it] = res["a"], res["b"], res["c"], res["Triaxiality"]
            Passage[it] = np.matrix(res["Passage"])
            P_inv[it] = np.linalg.inv(Passage[it]) # inverse passage between 2 consecutive frames
            # I multiply the position by the inverse passage matrix, 
            # in order to go in the natural frame of the ellipsoid
            pos = np.array((np.dot(P_inv[it], np.matrix(pos).T))).T
            # global inverse passage matrice
            Passage_gen = np.dot(Passage[it], Passage_gen) 
            P_inv_gen = np.dot(P_inv[it], P_inv_gen) 
        # I put the particles position in a matrix 3*N_particles
        dic = {"a_arr": a_arr, "b_arr": b_arr, "c_arr": c_arr,
               "Sphericity": Sphericity, "Elongation": Elongation, "Triaxiality": Triaxiality,
               "Passage": Passage, "P_inv": P_inv, "Passage_gen": Passage_gen, "P_inv_gen": P_inv_gen,
               "r_ell": r_ell, "pos": pos}
        return dic
    
    def compute_SQT_cv(self, pos: np.ndarray[float],
                       ) -> dict[str, np.ndarray[float]]:
        """
        Computes the shape tensor parameters by using convergence criteria.
        The method compute_SQT_once is called as long as the differences between Ss and Qs
        self.it_init calls is larger than self.error. 
        Afer self.it_loop without convergence, the error is increased until to reach convergence.
        # input:
        - pos: np.ndarray[float], 3d particles positions in the object
        # output:
        - a_arr, b_arr, c_arr: np.ndarray[float], shape=self.it_init+it, 
        semi principal axis length at each iteration
        - Sphericity, Elongation, Triaxiality: np.ndarray[float], shape=self.it_init+it,
        shape coefficients at each iteration
        - Passage, P_inv: np.ndarray[float], shape=(self.it_init+it, 3, 3)
        passage and inverse passage matrix at each iteration
        - Passage_gen, P_inv_gen: np.ndarray[float], shape=(3, 3)
        gloabal passage and inverse passage matrix (combined all iterations)
        - N_particles_initial: int, number of particles in the input
        - N_particles_final: int, number of particles in the final converged object
        - error_threshold_used: float, worst (i.e. largest) error used
        - self.it_init: int, number of iterations at the initialization and for the Ss and Qs mean
        - it_loop_fixed: int, every it_loop_fixed I increase the error allowed for convergence
        - total_iteration: int, total number of iteration needed for the full computation
        - computation_succes: int, either -1 (failed) or 0 (success)
        or +1 (succes but with larger error than minimum)
        """
        computation_succes = 0 # can also be +1 or -1
        N_particles_initial = len(pos)
        res = self.SQT_initialization(pos=pos, N_particles=N_particles_initial)
        Sphericity, Elongation = res["Sphericity"], res["Elongation"]
        Triaxiality = res["Triaxiality"]
        pos = res["pos"]
        r_ell = res["r_ell"]
        a_arr, b_arr, c_arr = res["a_arr"], res["b_arr"], res["c_arr"]
        Passage, P_inv = res["Passage"], res["P_inv"]
        Passage_gen, P_inv_gen = res["Passage_gen"], res["P_inv_gen"]
        it = 0
        err_Sphericity = np.std(Sphericity[-self.it_init:])/np.mean(Sphericity[-self.it_init:])
        err_Elongation = np.std(Elongation[-self.it_init:])/np.mean(Elongation[-self.it_init:])
        while ((err_Sphericity > self.error) or (err_Elongation > self.error)):
            ind_ell = np.where(r_ell < a_arr[-1])[0]
            N_particles_ellipse = len(ind_ell)
            # I compute the shape, here a>b>c have values in unit of Rvir, a can be different from 1
            res_once = self.compute_SQT_once(pos=pos[ind_ell], radius=r_ell[ind_ell],
                                             N_particles=N_particles_ellipse)
            Sphericity = np.append(Sphericity, res_once["Sphericity"])
            Elongation = np.append(Elongation, res_once["Elongation"])
            Triaxiality = np.append(Triaxiality, res_once["Triaxiality"])
            a_arr = np.append(a_arr, res_once["a"])
            b_arr = np.append(b_arr, res_once["b"])
            c_arr = np.append(c_arr, res_once["c"])
            Passage = np.append(Passage, res_once["Passage"].reshape((1,3,3)), axis=0)
            P_inv = np.append(P_inv, np.linalg.inv(Passage[-1]))
            pos = np.array((np.dot(P_inv[-1], np.matrix(pos.T)))).T
            r_ell = np.sqrt(pos[:,0]**2 + (pos[:,1]/Elongation[-1])**2 + (pos[:,2]/Sphericity[-1])**2)
            P_inv_gen = np.dot(P_inv[-1], P_inv_gen)
            Passage_gen = np.dot(Passage[-1], Passage_gen)
            err_Sphericity = np.std(Sphericity[-self.it_init:])/np.mean(Sphericity[-self.it_init:])
            err_Elongation = np.std(Elongation[-self.it_init:])/np.mean(Elongation[-self.it_init:])
            it += 1
            # if I do not reach the error threshold after it_loop
            if it > self.it_loop:
                computation_succes = 1
                self.error *= 1.1 # I increase the error threshold
                # then I increase the number of loops threshold
                self.it_loop += self.it_loop_fixed
            # if I have not enough particles within the halo
            if N_particles_ellipse < self.N_part_frac * N_particles_initial:
                print("not enough particles")
                computation_succes = -1
                # then I stop
                break
        res = {"a_arr": a_arr, "b_arr": b_arr, "c_arr": c_arr,
               "Sphericity": Sphericity, "Elongation": Elongation, "Triaxiality": Triaxiality,
               "Passage": Passage, "P_inv": P_inv, "Passage_gen": Passage_gen, "P_inv_gen": P_inv_gen,
               "N_particles_initial": N_particles_initial, "N_particles_final": N_particles_ellipse,
               "error_threshold_used": self.error, "it_init_used": self.it_init,
               "it_loop_fixed": self.it_loop_fixed, "total_iteration": it+self.it_init,
               "computation_succes": computation_succes,}
        return res
    
class test_sphericity(Smooth_halo, Sphericity):
    """
    This class allows to test the shape computation of a halo
    """

    def __init__(self,):
        self.a_ax_th: float = 1
        self.b_ax_th: float = 0.8
        self.c_ax_th: float = 0.2
        self.S_th: float = self.c_ax_th/self.a_ax_th
        self.Q_th: float = self.b_ax_th/self.a_ax_th
        if self.a_ax_th**2 - self.c_ax_th**2 != 0:
            self.T_th: float = (self.a_ax_th**2 - self.b_ax_th**2)/(self.a_ax_th**2 - self.c_ax_th**2)
        else :
            self.T_th = np.nan
        self.ellipsoidal_kind: str = "ellipsoid"
        self.it_init: int = 30
        self.it_loop: int = 100
        self.it_loop_fixed: int = 100
        self.error: float = 0.001
        self.N_part_frac: float = 0.1
        self.N_halos: int = 100
        self.N_part_th: int = 1000
        self.N_bin_th: int = 30
        self.R_max_th: float = 1/self.c_ax_th
        self.res_th: float = 0.01

    def rotate(self, pos: np.ndarray[float],
               theta_x: float,
               theta_y: float,
               theta_z: float,
               ) -> np.ndarray[float]:
        """ 
        change the orientation of the halo
        """
        # https://fr.wikipedia.org/wiki/Matrice_de_rotation#En_dimension_trois
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        pos_m = np.matrix(np.array((x,y,z)))
        cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
        R_x = np.matrix([[1,0,0],[0,cos_x,-sin_x],[0,sin_x,cos_x]])
        cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
        R_y = np.matrix([[cos_y,0,sin_y],[0,1,0],[-sin_y,0,cos_y]])
        cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
        R_z = np.matrix([[cos_z,-sin_z,0],[sin_z,cos_z,0],[0,0,1]])
        pos_m = np.dot(R_x, np.dot(R_y, np.dot(R_z, pos_m)))
        pos_new = np.array((pos_m)).transpose()
        return pos_new

    def test_on_many_halos(self,) -> dict[str, float]:
        S_save, Q_save, T_save = np.zeros((self.N_halos)), np.zeros((self.N_halos)), np.zeros((self.N_halos))
        N_part_beg_save, N_part_end_save = np.zeros((self.N_halos), dtype=int), np.zeros((self.N_halos), dtype=int)
        iterations_save = np.zeros((self.N_halos), dtype=int)
        errors_save = np.zeros((self.N_halos))
        angles = np.random.random((self.N_halos, 3)) * 2 * np.pi
        for h in range(self.N_halos):
            my_halo = self.smooth_halo_creation(N_part=self.N_part_th, N_bin=self.N_bin_th,
                                                R_max=self.R_max_th, res=self.res_th,
                                                a_ax=self.a_ax_th, b_ax=self.b_ax_th, c_ax=self.c_ax_th)
            pos = my_halo["data"]
            #self.beauty_plot_colorbar(pos=pos, ax_2=2)
            # rotate the halo
            pos = self.rotate(pos=pos, theta_x=angles[h,0], theta_y=angles[h,1], theta_z=angles[h,2])
            #self.beauty_plot_colorbar(pos=pos, ax_2=2)
            res = self.compute_SQT_cv(pos=pos)
            S_save[h], Q_save[h], T_save[h] = res["Sphericity"][-1], res["Elongation"][-1], res["Triaxiality"][-1]
            N_part_beg_save[h], N_part_end_save[h] = res["N_particles_initial"], res["N_particles_final"]
            errors_save[h] = res["error_threshold_used"]
            iterations_save[h] = res["total_iteration"]
        S_mean, S_std = np.mean(S_save), np.std(S_save)
        Q_mean, Q_std = np.mean(Q_save), np.std(Q_save)
        T_mean, T_std = np.mean(T_save), np.std(T_save)
        N_part_beg_save_mean = np.mean(N_part_beg_save)
        N_part_end_save_mean = np.mean(N_part_end_save)
        errors_save_mean = np.mean(errors_save)
        iterations_save_mean = np.mean(iterations_save)
        err_S = (S_mean - self.S_th)/S_std
        err_Q = (Q_mean - self.Q_th)/Q_std
        err_T = (T_mean - self.T_th)/T_std
        res = {"S_mean": S_mean, "S_std": S_std,
               "Q_mean": Q_mean, "Q_std": Q_std,
               "T_mean": T_mean, "T_std": T_std,
               "err_S": err_S, "err_Q": err_Q, "err_T": err_T,
               "N_part_beg_save_mean": N_part_beg_save_mean,
               "N_part_end_save_mean": N_part_end_save_mean,
               "errors_save_mean": errors_save_mean,
               "iterations_save_mean": iterations_save_mean,
               }
        return res




if __name__ == '__main__':
    my_test = test_sphericity()
    #my_test.N_halos = 2
    res = my_test.test_on_many_halos()
    print(res)
