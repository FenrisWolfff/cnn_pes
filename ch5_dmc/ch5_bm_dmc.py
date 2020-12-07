#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyvibdmc as dmc
from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *
from pyvibdmc import potential_manager as pm

if __name__ == '__main__': 
    pot_dir = 'BowmanCH5Pot/'
    py_file = 'ch5_pot.py'
    pot_func = 'rjd_ch5'

    ch5_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=8)


    ch5_coord_raw = np.array([[ 3.64370000e-05, 2.61250000e-04, 2.05490000e-05],
                    [ 6.96823202e-02, 1.10553262e+00, 2.26690000e-05],
                    [ 9.45749638e-01, -7.32895080e-01, -6.79440001e-05],
                    [ 1.18207493e+00, 1.87991209e-01, 9.14499999e-06],
                    [-4.36525904e-01, -3.33258609e-01, -9.39243426e-01],
                    [-4.36588351e-01, -3.33182565e-01, 9.39259010e-01]])
    ch5_coord = Constants.convert(ch5_coord_raw, 'angstroms', to_AU=True)*1.1
    print(ch5_coord)
    atoms = ['C','H','H','H','H','H']
    
    for sim_num in range(10):
        myDMC = dmc.DMC_Sim(sim_name=f"ch5_bm_{sim_num}",
                              output_folder="ch5_bm_dmc_output",
                              weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                              num_walkers=20000, #number of geometries exploring the potential surface
                              num_timesteps=10000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                              equil_steps=500, #how long before we start collecting wave functions
                              chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                              wfn_every=1000, #collect a wave function every "wfn_every" time steps
                              desc_wt_steps=100, #number of time steps you allow for descendant weighting per wave function
                              atoms=atoms,
                              delta_t=1, #the size of the time step in atomic units
                              potential=ch5_pot,
                              start_structures=np.expand_dims(ch5_coord,axis=0), #can provide a single geometry, or an ensemble of geometries
                              masses=None, #can put in artificial masses, otherwise it auto-pulls values from the atoms string
        )
        myDMC.run()

# In[ ]:




