#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from webgeocalc import StateVector
import pandas as pd

KERNEL="VOYAGER/kernels/spk/vgr2_ura083.bsp"
KERNEL_SETS=[1, 2, 3]  # Solar System, leapseconds, planetary constants
times=np.loadtxt("wgc_times_all.txt", dtype=str)

sv = StateVector(
    kernels=KERNEL_SETS,
    kernel_paths=KERNEL,
    times=list(times),
    target="VOYAGER 2",
    observer="URANUS",
    reference_frame="IAU_URANUS",
    aberration_correction="NONE",
    state_representation="SPHERICAL",
    verbose=True,
)
r = sv.run()

# Converstion to U1 coordinates
#------------------------------

COLAT = np.asarray(r['COLATITUDE'])
LON   = np.asarray(r['LONGITUDE'])
RAD   = np.asarray(r['RADIUS'])

theta = (180 - COLAT) * np.pi / 180

# Find closest approach to Uranus
idx = np.argmin(r['RADIUS'])

U1_closest_approach = 302
IAU_closest_approach = r['LONGITUDE'][idx]

#Move longitude to U1 frame
phi_U1 = (LON - IAU_closest_approach + U1_closest_approach) % 360 # This is west longitude
phi_U1 = (360 - phi_U1) % 360 # Convert to east longitude
phi_U1 *= np.pi / 180

df = pd.DataFrame({
    'PDSTIME': times,
    'radius': RAD,
    'colatitude': theta,
    'longitude': phi_U1,
})

df.to_csv("vg2_positions_1_92s.csv", index=False)