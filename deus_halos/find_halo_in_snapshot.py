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
from functools import lru_cache
from tqdm import tqdm
from simu_param import Simu_param
class Find_halo_in_snapshot(Simu_param):

    def __init__(self, cosmo: str, z: float, N_cubes: int, n_dim: int,) -> None:
        super(Find_halo_in_snapshot, self).__init__(cosmo, z)
        self.N_cubes = N_cubes  # number of snapshot file
        self.n_dim = n_dim  # number of dimension (it should be 3)
        # number of cubs in 1 direction
        self.N_cubes_1D = int(np.power(self.N_cubes, 1/self.n_dim)) + 1
        #  boundaries is a 1D np.array containing the bin_edges (in Mpc/h) of the full box cut in N_cubes small cubes.
        #  As the simulation is a cube, it is sufficient to have it in 1D, and to re-use it in all directions i
        self.boundaries = np.linspace(0,
                                      self.L_BOX,
                                      self.N_cubes_1D+1,
                                      dtype=float)

    def check_halo_in_which_cube(self,
                                 cdm: np.ndarray[float],
                                 Rlim: np.ndarray[float],
                                 dim: int | None = None,
                                 halo_cube: np.ndarray[int] | None = None,
                                 ) -> np.ndarray[int]:
        ''' 
        This functions finds in which cube (i.e. subpart) of the full simulation box
        the halo centers cdm are. It also finds if other cubes are needed in order to extract the halo
        if the halo is close (i;e. less distant than Rlim) to a boundary of a cube.
        It does this for all haloes. 
        ###########################################################################
        # input:
        - cdm, np.ndarray[float], shape=(N_haloes, 3): cdm (in Mpc/h) from halo files of DEUS
        - Rlim, np.ndarray[float], shape=(N_haloes): factor * Rvir (in Mpc/h)
        - dim, int | None, optional, dimension considred (i.e. dim=1 <=> x, dim=2 <=> y)
        - halo_cube, np.ndarray[int] | None, optional, shape=(N_haloes, 2*self.n_dim+1)
        contains the halo numero, the self.n_dim cube positions containing the halo 
        in the range[0, self.N_cubes_1D-1] (1 integer for each direction x, y, ...)
        and the self.n_dim indicators if the halo is close to the boundary and if yes in which sense
        i_x, i_y and i_z, integers = -1, 0, 1 and e.g. for the x direction:
        If i_x = 0, it means that cdm_i = is far from the boundary of its cube on the x direction
        If i_x = 1, it means that cdm_i + Rlim > upper boundary of its cube on the x direction
        If i_x = -1, it means that cdm_i - Rlim < lower boundary of its cube on the x direction
        Note that halo_cube is also what this function returns (recursive function)
        ##########################################################################
        # output:
        - halo_cube (recursive function), see last input
        '''
        # First I check if the center is inside the simulation box [0,L_box],
        # and I set it if it is not the case (the simulation must used periodic condition)
        if dim is None:
            dim = 1
        print("dim =", dim)
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
            bound_min = self.boundaries[c_i]
            bound_max = self.boundaries[c_i+1]
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
                                                      dim=dim+1,
                                                      halo_cube=halo_cube,
                                                      )

        return halo_cube

    def compute_halo_cube_num(self,
                              c_x: int,
                              c_y: int,
                              c_z: int,
                              ) -> int:
        """
        Computes the deus snapshot binary file numero from the cubes in the 3 directions (x, y and z)
        This numero is between [0, self.N_cubes-1] and is given by
        the formula below
        """
        # numero of the file corresponding to 1 cube c_x, c_y, c_z, integers between [0,N_cubes-1]
        return c_x * self.N_cubes_1D**2 + c_y * self.N_cubes_1D + c_z

    def set_to_dic(self,
                   c_x: int,
                   c_y: int,
                   c_z: int,
                   i_x: int,
                   i_y: int,
                   i_z: int,
                   ) -> dict[str, int]:
        """
        set the integers characterizing the halo position within the simulation box
        inside a dictionary
        """
        return {"cube_x": c_x,
                "cube_y": c_y,
                "cube_z": c_z,
                "shift_category_x": i_x,
                "shift_category_y": i_y,
                "shift_category_z": i_z,
                "halo_cube_num": self.compute_halo_cube_num(c_x=c_x,
                                                            c_y=c_y,
                                                            c_z=c_z)
                }

    def find_cubes(self,
                   halo_cube: np.ndarray[int],
                   ) -> dict[str, int | dict[str, int]]:
        '''
        Characterizes the (between 1 and 2**self.n_dim) cube(s) (i.e. deus binary snapshot file)
        that contains all the particles needed to study a halo.
        It describes also if these cubes are at the opposite of the full simulation box 
        (see for more details)
        ############################################################################
        # input:
        - halo_cube: output of check_halo_in_which_cube function
        # output:
        - halo_dic, dict[str, int | dict[str, int]]: characterizes the cubes containing the 
        halo particles. This dictionnary contains 3 integers, the halo numero h,
        the indicator of the number of cubes needed n (between [0, self.n_dim])
        and the number of cubes needed (2**n between [1, 2**self.n_dim]).
        It also contains between 1 and 2**self.n_dim dict[str, int], 
        each dictionnary characterizing 1 cube that contains particles of the halo.
        Each dictionnary contains 2 * self.n_dim + 1 keys and values.
        The terms are the output of the compute_halo_cube_num function, 
        the self.n_dim cube positions containing the halo in the range
        [0, self.N_cubes_1D-1] (1 integer for each direction x, y, ...)
        and self.n_dim integers, 1 by direction x, y and z, indicating if
        there the cube considered is shifted wrt the cube where the halo center is
        (in this case the integer is either -1, 0 or +1) or not 
        (in this case the integer is  = 137). In the shifted case, we have 3 cases for our cube:
            - 0: means that there is this cube and the main cube are touching each other
            - +1 means that the halo center is close to self.L_BOX while this cube position
         is close to 0
            - -1 means that the halo center is close to 0 while this cube position is close to 
        self.L_BOX
        '''
        # n_cube_ind characterizes the number of cubes where I need to look for particles
        n_cube_ind = 0  # number of direction I need to shift, to look for in another cube
        # the number 137 means that nothing needs to be done in the i direction,
        N_fun = 137  # N_fun could be anything not between [-1,1]
        h = halo_cube[0]  # halo numero
        # cube numero where the halo center cdm_i is in the i direction
        c_x, c_y, c_z = halo_cube[1], halo_cube[2], halo_cube[3]
        # need to look for another cube in the i direction (i_n=-1 or 1) or not (i_n=0)
        x_n, y_n, z_n = halo_cube[4], halo_cube[5], halo_cube[6]
        # I begin by considering the cube of the halo center, initialization
        # if all the i_n=0, I just add n_cube_ind and return the dictionary
        # no shift case, 1 cube
        halo_dic = {}
        halo_dic["halo_numero"] = h
        halo_dic["halo_cube_main"] = self.set_to_dic(c_x=c_x,
                                                     c_y=c_y,
                                                     c_z=c_z,
                                                     i_x=N_fun,
                                                     i_y=N_fun,
                                                     i_z=N_fun)
        # potentially 7 more cubes than the one of the halo center, 1 cube by if
        if x_n != 0:  # at least 2 cubes (shift on x), but maybe more
            # cube numero of the cube neighbour in the i=1 direction (for x)
            c_x_new = (c_x + x_n) % self.N_cubes_1D
            # The i_n_new's won't be used in the direction where i_n=0, and I will use N_fun in that direction
            # if used, it is =0 if the cube containing the halo center is not close to the boundary
            # of the full simulation box. Otherwise, it is =-1 or =1.
            x_n_new = (c_x + x_n) // self.N_cubes_1D
            halo_dic["halo_cube_x"] = self.set_to_dic(c_x=c_x_new,
                                                      c_y=c_y,
                                                      c_z=c_z,
                                                      i_x=x_n_new,
                                                      i_y=N_fun,
                                                      i_z=N_fun)
            n_cube_ind += 1  # I need to shift in the x direction
        if y_n != 0:  # at least 2 cubes (shift on y), but maybe more
            c_y_new = (c_y + y_n) % self.N_cubes_1D
            y_n_new = (c_y + y_n) // self.N_cubes_1D
            halo_dic["halo_cube_y"] = self.set_to_dic(c_x=c_x,
                                                      c_y=c_y_new,
                                                      c_z=c_z,
                                                      i_x=N_fun,
                                                      i_y=y_n_new,
                                                      i_z=N_fun)
            n_cube_ind += 1  # I need to shift in the y direction
        if z_n != 0:  # at least 2 cubes (shift on z), but maybe more
            c_z_new = (c_z + z_n) % self.N_cubes_1D
            z_n_new = (c_z + z_n) // self.N_cubes_1D
            halo_dic["halo_cube_z"] = self.set_to_dic(c_x=c_x,
                                                      c_y=c_y,
                                                      c_z=c_z_new,
                                                      i_x=N_fun,
                                                      i_y=N_fun,
                                                      i_z=z_n_new)
            n_cube_ind += 1  # I need to shift in the z direction
        # if I need more than 1 cube, I also need to consider the shift in both directions
        # at least 4 cubes (shift on x and y), but maybe more
        if x_n != 0 and y_n != 0:
            halo_dic["halo_cube_xy"] = self.set_to_dic(c_x=c_x_new,
                                                       c_y=c_y_new,
                                                       c_z=c_z,
                                                       i_x=x_n_new,
                                                       i_y=y_n_new,
                                                       i_z=N_fun)
        # at least 4 cubes (shift on x and z), but maybe more
        if x_n != 0 and z_n != 0:
            halo_dic["halo_cube_xz"] = self.set_to_dic(c_x=c_x_new,
                                                       c_y=c_y,
                                                       c_z=c_z_new,
                                                       i_x=x_n_new,
                                                       i_y=N_fun,
                                                       i_z=z_n_new)
        # at least 4 cubes (shift on y and z), but maybe more
        if y_n != 0 and z_n != 0:
            halo_dic["halo_cube_yz"] = self.set_to_dic(c_x=c_x,
                                                       c_y=c_y_new,
                                                       c_z=c_z_new,
                                                       i_x=N_fun,
                                                       i_y=y_n_new,
                                                       i_z=z_n_new)
        # 8 cubes (maximum) (shift on x, y and z)
        if x_n != 0 and y_n != 0 and z_n != 0:
            halo_dic["halo_cube_xyz"] = self.set_to_dic(c_x=c_x_new,
                                                        c_y=c_y_new,
                                                        c_z=c_z_new,
                                                        i_x=x_n_new,
                                                        i_y=y_n_new,
                                                        i_z=z_n_new)
        halo_dic["n_cubes_needed"] = 2**n_cube_ind
        halo_dic["n_cubes_needed_indicator"] = n_cube_ind
        return halo_dic

    def check_if_file_exist(self,
                            path_cube: str,
                            name_file: str,
                            halo_cube: dict[str, int | dict[int]],
                            ) -> dict[str, bool]:
        """
        check if the deus snapshot files corresponding to the cubes of halo_cube
        for a halo are present at path+name_file
        """
        file_existence = {}
        for cube_name, my_cube in halo_cube.items():
            if type(my_cube) == dict:
                if isfile(path_cube + name_file + '_' +
                          str('{:d}'.format(my_cube["halo_cube_num"]).zfill(5))):
                    file_existence["file_exist_cube_" +
                                   cube_name] = True  # the file c exists
                else:
                    file_existence["file_exist_cube_" +
                                   cube_name] = False
        return file_existence

    @lru_cache
    def create_df_old(self,
                      df,
                      halo_cube: dict[str, int | dict[str, int]],
                      loc: int,
                      ):
        print("loc = ", loc)
        halo_dic = self.find_cubes(halo_cube=halo_cube[loc])
        for key, val in halo_dic.items():
            if type(val) != dict:
                if key not in df.columns:
                    df[key] = np.nan
                df.iloc[loc][key] = val
            else:
                for cube_name, cube_val in val.items():
                    if key + cube_name not in df.columns:
                        df[key + cube_name] = np.nan
                    df.iloc[loc][key + cube_name] = cube_val
        if loc < len(halo_cube) - 1:
            df = self.create_df_old(df=df, halo_cube=halo_cube, loc=loc+1)
        return df

    def create_df(self,
                  halo_cube: np.ndarray[int],
                  all_halo: pd.DataFrame | None = None,
                  ) -> pd.DataFrame:
        if all_halo is None:
            all_halo = pd.DataFrame()
        my_cubes = self.find_cubes(halo_cube=halo_cube)
        non_dic = {}
        for key, val in my_cubes.items():
            if type(val) != dict:
                non_dic[key] = val
        one_halo = pd.Series(non_dic)
        for key, val in my_cubes.items():
            if type(val) == dict:
                one_halo = pd.concat([one_halo, pd.Series(val.values(),
                                                          index=[key + "_" + k for k in val.keys()])])
        one_halo = pd.DataFrame(one_halo).T
        return pd.concat([all_halo, one_halo], axis=0)

    def create_df_final(self,
                        all_halo: pd.DataFrame,
                        my_cubes: dict[str, int | dict[str, int]],
                        ) -> pd.DataFrame:
        one_halo = self.create_serie(my_cubes=my_cubes)
        return pd.concat([all_halo, one_halo], axis=0)


if __name__ == '__main__':
    # see also periodicity
    # load halo file with their cdm and N_part_fof
    path = "./data/z_0/rpcdm/halos_position/halos_position/"
    file_name = "../halo_properties/halo_pos.dat"
    halo = pd.read_csv(path + file_name)
    cdm = halo[["x (Mpc/h)", "y (Mpc/h)", "z (Mpc/h)"]].values
    N_FOF_all = halo["halo N_part"].values

    # instantiate my object
    find_halo = Find_halo_in_snapshot(cosmo="rpcdm", z=0, N_cubes=512, n_dim=3)

    # estimate Rvir then Rlim
    Rvir = np.power(N_FOF_all * 3/(4 * np.pi *
                    find_halo.RHO_VIR_BRIAN_NORMAN), 1/find_halo.n_dim)
    factor = 5
    Rlim = factor * Rvir

    # first characterization of halo
    halo_cube = find_halo.check_halo_in_which_cube(cdm=cdm,
                                                   Rlim=Rlim)
    # to check if the deus files needed exist or not
    path_cube = "./data/z_0/rpcdm/snapshot/"
    name_file = "fof_boxlen648_n2048_rpcdmw7_cube"
    import time
    t_beg = time.time()
    print(len(halo_cube))
    all_halo = pd.DataFrame()
    for f in tqdm(range(len(halo_cube))):  # len(halo_cube)):
        # second characterization of the halos
        all_halo = find_halo.create_df(
            all_halo=all_halo,
            halo_cube=halo_cube[f])
        # file_existence = find_halo.check_if_file_exist(path_cube=path_cube,
        #                                               name_file=name_file,
        #                                               halo_cube=my_cubes_f)
    t_end = time.time()
    print(all_halo.head())
    print(all_halo.shape)
    all_halo.to_csv(path + "../halo_properties/halo_cube_pos.dat",
                    index=False)
    print(t_end-t_beg)
