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


# In[ ]:


if __name__ == '__main__': #if using multiprocessing on windows / mac, you need to encapsulate using this line
    pot_dir = 'sample_potentials/FortPots/Partridge_Schwenke_H2O/' #this directory is part of the one you copied that is outside of pyvibdmc.
    py_file = 'h2o_potential.py'
    pot_func = 'water_pot' # def water_pot(cds) in h2o_potential.py

    #The Potential object assumes you have already made a .so file and can successfully call it from Python
    water_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=8)

    #optional num_cores parameter for multiprocessing, should not exceed the number of cores on the CPU
    #your machine has. Can use multiprocessing.cpu_count()

    # Starting Structure
    # Equilibrium geometry of water in *atomic units*, then blown up by 1.01 to not start at the bottom of the potential.
    water_coord = np.array([[1.81005599,  0.        ,  0.        ],
                           [-0.45344658,  1.75233806,  0.        ],
                           [ 0.        ,  0.        ,  0.        ]]) * 1.01

    for sim_num in range(5):
        myDMC = dmc.DMC_Sim(sim_name=f"h2o_nn_{sim_num}",
                              output_folder="h2o_ps_dmc_output",
                              weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                              num_walkers=20000, #number of geometries exploring the potential surface
                              num_timesteps=10000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                              equil_steps=500, #how long before we start collecting wave functions
                              chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                              wfn_every=1000, #collect a wave function every "wfn_every" time steps
                              desc_wt_steps=100, #number of time steps you allow for descendant weighting per wave function
                              atoms=['H','H','O'],
                              delta_t=1, #the size of the time step in atomic units
                              potential=water_pot,
                              start_structures=np.expand_dims(water_coord,axis=0), #can provide a single geometry, or an ensemble of geometries
                              masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string
        )
        myDMC.run()


# In[ ]:




