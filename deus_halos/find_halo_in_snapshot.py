#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/12/27

@author: guillaume
"""

"""
This script allows to identify inside which snapshot a halo lies
"""




import numpy as np
import pandas as pd
from os.path import isfile
from simu_param import Simu_param
class Find_halo_in_snapshot(Simu_param):

    def __init__(self, *args, **kwargs):
        super(Find_halo_in_snapshot, self).__init__(*args, **kwargs)
        self.N_cubes = 512
        self.n_dim = 3
        self.N_cubes_1D = int(np.power(self.N_cubes, 1/self.n_dim))

    def check_halo_in_which_cube(self,
                                 cdm: np.ndarray[float],
                                 Rlim: np.ndarray[float],
                                 dim: int = None,
                                 halo_cube: np.ndarray[float] = None,
                                 boundaries: np.ndarray[float] = None,
                                 ) -> np.ndarray[float]:
        ''' This function finds in which cube of the full simulation box are 
        the halo centers of all haloes in the direction i (x <=> i=1, y <=> i=2, z <=> i=3). 
        It also finds if those centers are close (according to Rlim) of the boundary 
        of the cubes. It returns halo_cube, described below.
        Be carefull, Rlim should be smaller than L_box/N_cubes_1D
        boundaries is a 1D np.array containing the bin_edges (in Mpc/h) of the full box cut in N_cubes small cubes.
        As the simulation is a cube, it is sufficient to have it in 1D, and to re-use it in all directions i
        cdm_i are all the halo centers in the direction i (in Mpc/h).
        I want to look for particles inside a cube centered on the halo center and of size 2*Rlim.
        Rlim is a np.array containing N_haloes distances (in Mpc/h).
        halo_cube is a np.array of N_haloes lines and 7 columns, made of integers.
        At the begining, it contains only zeros, then it is filled on each direction i
        The first column contains the halo numero h between [0,N_haloes-1]
        The second to fourth columns (the c_i's) fully characterize the cube which 
        contains the halo center, one for each direction i. The c_i are between [0,N_cubes_1D-1],
        where N_cubes_1D=(N_cubes)**(1/3) (N_cubes_1S is the number of cubes in 1D).
        The fifth to seventh columns (the i_n's) characterizes if the halo center is 
        closed of the boundary of its cube, one for each direction i.
        The i_n's are between [-1,1], and
        If i_n = 0, it means that cdm_i = is far from the boundary of its cube on the i direction
        If i_n = 1, it means that cdm_i + Rlim > upper boundary of its cube on the i direction
        If i_n = -1, it means that cdm_i - Rlim < lower boundary of its cube on the i direction
        '''
        # First I check if the center is inside the simulation box [0,L_box],
        # and I set it if it is not the case (the simulation must used periodic condition)
        if dim is None:
            dim = 1
        print("dim =", dim)
        if boundaries is None:
            boundaries = np.linspace(0,
                                     self.L_BOX,
                                     self.N_cubes_1D+1,
                                     dtype=float)
        if halo_cube is None:
            halo_cube = np.zeros((len(cdm), self.n_dim*2 + 1),
                                 dtype=int)
        cdm_i = cdm[:, dim-1]
        ind_i_plus = np.where(cdm_i > self.L_BOX)[0]
        cdm_i[ind_i_plus] = cdm_i[ind_i_plus] - self.L_BOX
        ind_i_minus = np.where(cdm_i < 0)[0]
        cdm_i[ind_i_minus] = cdm_i[ind_i_minus] + self.L_BOX
        # loop on the cubes inside the full box in the i direction
        for c_i in range(self.N_cubes_1D):
            # boundaries of the cube c_i in the i direction
            bound_min = boundaries[c_i]
            bound_max = boundaries[c_i+1]
            ind_cdm_i_sup = np.where(cdm_i > bound_min)[0]
            ind_cdm_i_inf = np.where(cdm_i[ind_cdm_i_sup] < bound_max)[0]
            # contains only haloes inside the cube c_i
            ind_i = ind_cdm_i_sup[ind_cdm_i_inf]
            cdm_i_used = cdm_i[ind_i]
            Rlim_used = Rlim[ind_i]
            N_c_i = len(ind_i)
            i_n_used = np.zeros((N_c_i), dtype=int)
            ind_plus = np.where(cdm_i_used - Rlim_used <= bound_min)[0]
            i_n_used[ind_plus] = -1  # lower boundary effect possible
            ind_minus = np.where(cdm_i_used + Rlim_used >= bound_max)[0]
            i_n_used[ind_minus] = 1  # upper boundary effect possible
            c_i_array = np.ones((N_c_i), dtype=int) * c_i
            halo_cube[ind_i, 0], halo_cube[ind_i,
                                           dim], halo_cube[ind_i, dim+3] = ind_i, c_i_array, i_n_used
        if dim < self.n_dim:
            halo_cube = self.check_halo_in_which_cube(cdm=cdm,
                                                      Rlim=Rlim,
                                                      halo_cube=halo_cube,
                                                      dim=dim+1,
                                                      boundaries=boundaries)

        return halo_cube

    def find_cubes(self,
                   halo_cube: np.ndarray[float],
                   ) -> list[list[int]]:
        '''This function takes as an argument what returns the function: 
        check_halo_in_which_cube(data_properties,boundaries,factor,L_box).
        It finds explicitly all the cubes (between 1 and 8) that are assigned to each halo 
        and if these cubes are at the opposite of the full simulation box.
        It returns halo, a list, containing (1 + N_cubes + 1) lists .
        The first list contains the halo numero (1 integer between [0,N_haloes-1]).
        The last list contains n, which is an integer between [0,3] characterizing 
        the number of cubes = 2**n I need to explore, in order to look for particles.
        Thus, they are 2**n lists in between, containg each 6 integers. 
        The 3 first are the c_i's', which characterized the cube I need to look for,
        they are between [0,7]. The 3 lasts are the i_n, which characterized if the cube 
        is at the opposite to the simulation box compared to the cube containg the halo center.
        They are between [-1,1] or = 137. i_n=137 means that in the direction i, 
        I do not change of cube compared to the halo center's cube. In the other cases,
        so between [-1,1], I do change of cube in the direction i. i_n=0 means that
        the cube changed and the original cube touchs each other in the simulation box.
        i_n=1 means that the halo center cdm_i is close to 1 and the cube position coordinates 
        (x's, y's and z's) are close to 0. i_n=-1 means the opposite. 
        '''
        # n will characterize the number of cubes where I need to look for particles
        n = 0  # number of direction I need to shift, to look for in another cube
        # the number 137 means that nothing needs to be done ine the i direction,
        N_fun = 137  # N_fun could be anything not between [-1,1]
        h = halo_cube[0]  # halo numero
        # cube numero where the halo center cdm_i is in the i direction
        c_x, c_y, c_z = halo_cube[1], halo_cube[2], halo_cube[3]
        # need to look for another cube in the i direction (i_n=-1 or 1) or not (i_n=0)
        x_n, y_n, z_n = halo_cube[4], halo_cube[5], halo_cube[6]
        # I begin by considering the cube of the halo center, initialization
        # if all the i_n=0, I just add [n] to that list and return it
        # no shift case, 1 cube
        halo_dic = {}
        halo_dic["halo_numero"] = h

        def set_to_dic(c_x, c_y, c_z, i_x, i_y, i_z): return {"cube_x": c_x,
                                                              "cube_y": c_y,
                                                              "cube_z": c_z,
                                                              "shift_category_x": i_x,
                                                              "shift_category_y": i_y,
                                                              "shift_category_z": i_z
                                                              }
        halo_dic["halo_cube_main"] = set_to_dic(
            c_x, c_y, c_z, N_fun, N_fun, N_fun)
        halo = [[h], [c_x, c_y, c_z, N_fun, N_fun, N_fun]]
        # potentially 7 more cubes than the one of the halo center, 1 cube by if
        if x_n != 0:  # at least 2 cubes (shift on x), but maybe more
            # cube numero of the cube neighbour in the i=1 direction (for x)
            c_x_new = (c_x + x_n) % self.N_cubes_1D
            # The i_n_new's won't be used in the direction where i_n=0, and I will use N_fun in that direction
            # if used, it is =0 if the cube containing the halo center is not close to the boundary
            # of the full simulation box. Otherwise, it is =-1 or =1.
            x_n_new = (c_x + x_n) // self.N_cubes_1D
            halo_dic["halo_cube_x"] = set_to_dic(
                c_x_new, c_y, c_z, x_n_new, N_fun, N_fun)
            halo = halo + [[c_x_new, c_y, c_z, x_n_new, N_fun, N_fun]]
            n += 1  # I need to shift in the x direction
        if y_n != 0:  # at least 2 cubes (shift on y), but maybe more
            c_y_new = (c_y + y_n) % self.N_cubes_1D
            y_n_new = (c_y + y_n) // self.N_cubes_1D
            halo_dic["halo_cube_y"] = set_to_dic(
                c_x, c_y_new, c_z, N_fun, y_n_new, N_fun)
            halo = halo + [[c_x, c_y_new, c_z, N_fun, y_n_new, N_fun]]
            n += 1  # I need to shift in the y direction
        if z_n != 0:  # at least 2 cubes (shift on z), but maybe more
            c_z_new = (c_z + z_n) % self.N_cubes_1D
            z_n_new = (c_z + z_n) // self.N_cubes_1D
            halo_dic["halo_cube_z"] = set_to_dic(
                c_x, c_y, c_z_new, N_fun, N_fun, z_n_new)
            halo = halo + [[c_x, c_y, c_z_new, N_fun, N_fun, z_n_new]]
            n += 1  # I need to shift in the z direction
        # if I need more than 1 cube, I also need to consider the shift in both directions
        # at least 4 cubes (shift on x and y), but maybe more
        if x_n != 0 and y_n != 0:
            halo_dic["halo_cube_xy"] = set_to_dic(
                c_x_new, c_y_new, c_z, x_n_new, y_n_new, N_fun)
            halo = halo + [[c_x_new, c_y_new, c_z, x_n_new, y_n_new, N_fun]]
        # at least 4 cubes (shift on x and z), but maybe more
        if x_n != 0 and z_n != 0:
            halo_dic["halo_cube_xz"] = set_to_dic(
                c_x_new, c_y, c_z_new, x_n_new, N_fun, z_n_new)
            halo = halo + [[c_x_new, c_y, c_z_new, x_n_new, N_fun, z_n_new]]
        # at least 4 cubes (shift on y and z), but maybe more
        if y_n != 0 and z_n != 0:
            halo_dic["halo_cube_yz"] = set_to_dic(
                c_x, c_y_new, c_z_new, N_fun, y_n_new, z_n_new)
            halo = halo + [[c_x, c_y_new, c_z_new, N_fun, y_n_new, z_n_new]]
        # 8 cubes (maximum) (shift on x, y and z)
        if x_n != 0 and y_n != 0 and z_n != 0:
            halo_dic["halo_cube_xyz"] = set_to_dic(
                c_x_new, c_y_new, c_z_new, x_n_new, y_n_new, z_n_new)
            halo = halo + \
                [[c_x_new, c_y_new, c_z_new, x_n_new, y_n_new, z_n_new]]
        halo_dic["n_cubes_needed"] = 2**n
        halo_dic["n_cubes_needed_indicator"] = n
        # return halo + [[n]]
        return halo_dic

    def check_if_file_exist(self,
                            path_cube: str,
                            name_file: str,
                            halo_cube: list[list[int]],
                            ) -> tuple[np.ndarray[int], int]:
        """
        check if the files corresponding to the cubes of halo_cube for a halo
        are present at path+name_file
        """
        # N_cube_loop is an integer between [1,8]
        N_cube_loop = int(2**halo_cube[-1][0])
        # 0 = the file does not exist
        file_existence = np.zeros(N_cube_loop, dtype=int)
        for c in range(N_cube_loop):
            my_cube = halo_cube[c+1]
            # c_x, c_y and c_z integers between [0,N_cube_1D-1]
            c_i = np.array((my_cube[0], my_cube[1], my_cube[2]))
            # cube numbers in 1 direction in the simulation box
            # numero of the file corresponding to 1 cube c_x, c_y, c_z, integer between [0,N_cubes-1]
            num = int(c_i[0] * self.N_cubes_1D**2 +
                      c_i[1] * self.N_cubes_1D + c_i[2])
            if c == 0:
                num_main = num
            file_c_ex = isfile(path_cube+name_file+'_' +
                               str('{:d}'.format(num).zfill(5)))
            if file_c_ex:
                file_existence[c] = 1  # the file c exists
        return file_existence, num_main


if __name__ == '__main__':

    # step 1:
    # check_halo_in_which_cube
    # argument: boundaries, cdm[0], Rlim, halo_cube, 1, L_box)
    # need to know the box size and how the snapshot are cut
    # and also an estimate of the halo position and size
    # need to apply check_halo_in_which_cube as many times there are dimensions (=> 3 times)

    # step 2:
    # find_cubes
    # argument: output of check_halo_in_which_cube
    # It finds explicitly all the cubes (between 1 and 8) that are assigned to each halo
    # and if these cubes are at the opposite of the full simulation box
    # I need to check the periodicity anyway

    # see also periodicity and check_if_file_exist

    path = "./data/z_0/rpcdm/halos_position/halos_position/"
    file_name = '../halo_properties/halo_pos.dat'

    halo = pd.read_csv(path + file_name)

    # simu = Simu_param(cosmo="rpcdm", z=0)

    find_halo = Find_halo_in_snapshot(cosmo="rpcdm", z=0)

    cdm = halo[["x (Mpc/h)", "y (Mpc/h)", "z (Mpc/h)"]].values
    N_FOF_all = halo["halo N_part"].values
    Rvir = np.power(N_FOF_all * 3/(4 * np.pi *
                    find_halo.RHO_VIR_BRIAN_NORMAN), 1/find_halo.n_dim)
    factor = 5
    Rlim = factor * Rvir
    halo_cube = find_halo.check_halo_in_which_cube(cdm=cdm,
                                                   Rlim=Rlim)

    path_cube = "./data/z_0/rpcdm/snapshot/"
    name_file = "fof_boxlen648_n2048_rpcdmw7_cube"
    for f in range(3):
        my_cubes_f = find_halo.find_cubes(halo_cube[f])
        print(my_cubes_f)
        file_existence, num_main = find_halo.check_if_file_exist(path_cube=path_cube,
                                                                 name_file=name_file,
                                                                 halo_cube=my_cubes_f)
        # N_cube_1D=find_halo.N_cubes_1D)
        print(file_existence, num_main)
        if np.min(file_existence) == 1:  # it means that all the needed files exist
            print('The needed files do exist')
        else:
            print("the needed file do not exist")
        # if len(my_cubes) > 3 or my_cubes[1][-3] != 137 or my_cubes[1][-2] != 137 or my_cubes[1][-1] != 137:
        #    print(my_cubes)

    """halo_cube = pd.DataFrame(halo_cube, columns=["halo numero",
                                                 "cube of halo center x",
                                                 "cube of halo center y",
                                                 "cube of halo center z",
                                                 "cube close of center x",
                                                 "cube close of center y",
                                                 "cube close of center z"])

    print(halo_cube["cube of halo center x"].value_counts(),
          halo_cube["cube of halo center y"].value_counts(),
          halo_cube["cube of halo center z"].value_counts())

    print(halo_cube["cube close of center x"].value_counts(),
          halo_cube["cube close of center y"].value_counts(),
          halo_cube["cube close of center z"].value_counts())

    """
