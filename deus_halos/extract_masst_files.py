#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 2023/12/21

@author: guillaume
"""

'''
This file extract the information contained in the masst files of DEUS
so the halos position, particle number and identities
'''




import struct
import numpy as np
import pandas as pd
from simu_param import Simu_param
class Extract_halo_pos(Simu_param):
    def extract_halo_pos(self,
                         N_cube: int,
                         path: str,
                         file_name: str,
                         to_save: bool = False,
                         verbose: int = 0,
                         ) -> None:
        prop_save = pd.DataFrame()
        for f in range(N_cube):
            # I open the binary file number f
            file = open(path+file_name + str("{:d}".format(f).zfill(5)),
                        "rb")
            file.read(4)  # number bites
            # I put the number of halos in the file f in the array N_halos
            N_halos = struct.unpack("<i", file.read(4))[0]
            if verbose:
                print("file =", f, "N_halos =", N_halos)
            prop = np.zeros((N_halos, 5))
            for h in range(N_halos):
                file.read(8)
                # identity of the halo h of the file f
                identity = struct.unpack("<q", file.read(8))[0]
                # number of particles in the halo h of the file f
                N_part = struct.unpack("<i", file.read(4))[0]
                # position in box unit of the halo h in the file f
                pos = np.array(struct.unpack("<" + str(3) + "f",
                                             file.read(4*3)))
                pos *= self.L_BOX
                prop[h, 0] = identity
                prop[h, 1] = N_part
                prop[h, 2:] = pos
            prop_df = pd.DataFrame(
                prop, columns=["halo identity", "halo N_part",
                               "x (Mpc/h)", "y (Mpc/h)", "z (Mpc/h)"])
            prop_df["file"] = f
            prop_save = pd.concat([prop_save, prop_df], axis=0)
        if to_save:
            prop_save.to_csv(
                path + '../halo_properties/halo_pos.dat', index=False)


extract_halo = Extract_halo_pos(cosmo="rpcdm", z=0)


N_cube = 512
path = "./data/z_0/rpcdm/halos_position/halos_position/"
file_name = "fof_boxlen648_n2048_rpcdmw7_masst_"
extract_halo.extract_halo_pos(N_cube=N_cube,
                              path=path,
                              file_name=file_name,
                              to_save=True)
