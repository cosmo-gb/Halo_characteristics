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
def extract_halo_pos(N_cube, path, file_name):
    for f in range(N_cube):
        print('f =', f)
        # I open the binary file number f
        file = open(path+file_name+'_'+str('{:d}'.format(f).zfill(5))+"", "rb")
        file.read(4)  # number bites
        # I put the number of halos in the file f in the array N_halos
        N_halos = struct.unpack('<i', file.read(4))[0]
        print('N_halos =', N_halos)
        prop_save = np.zeros((N_halos, 5))
        identity_save = np.zeros((N_halos), dtype=int)
        for h in range(N_halos):
            file.read(8)
            identity = struct.unpack('<q', file.read(8))[0]
            identity_save[h] = identity
            # print('halo identity =',identity)
            # number of particles in the halo h of the file f
            N_part = struct.unpack('<i', file.read(4))[0]
            # print('N_part =',N_part)
            pos = np.array(struct.unpack(
                '<'+str(3)+'f', file.read(4*3))) * L_box
            prop_save[h, 0] = identity
            prop_save[h, 1] = N_part
            prop_save[h, 2:] = pos
        prop_save = pd.DataFrame(
            prop_save, columns=['halo identity', 'halo N_part', 'x (Mpc/h)', 'y', 'z'])
        prop_save.to_csv(path+'../halo_properties/halo_pos_' +
                         str(f)+'.dat', index=False)
        identity_save = pd.DataFrame(identity_save, columns=['halo identity'])
        identity_save.to_csv(
            path+'../halo_properties/halo_id_'+str(f)+'.dat', index=False)
    return ()
