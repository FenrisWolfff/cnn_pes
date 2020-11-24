import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *

from dscribe.descriptors import SOAP

from ase.io import read
from ase.build import molecule
from ase import Atoms



def soap_it(cds):
    #think this should be done in angstroms, so making it that way.
    structures = [Atoms(symbols=["H", "H", "O"], positions=cd) for cd in cds]
    species = ["H", "O"]
    rcut = 6.0
    nmax = 8
    lmax = 6

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax
    )
    feature_vectors = soap.create(structures, n_jobs=1)
    ryans_bullshit = np.reshape(feature_vectors, (len(cds), len(feature_vectors) // len(cds), -1))
    print('hello')


