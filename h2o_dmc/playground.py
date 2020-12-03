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
    water_coord = np.array([[[1.81005599,  0.        ,  0.        ],
                           [-0.45344658,  1.75233806,  0.        ],
                           [ 0.        ,  0.        ,  0.        ]]]) * 1.5

    print(water_pot.getpot(water_coord))
    
