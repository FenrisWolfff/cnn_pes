import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *

from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from ase import Atoms


def acsf_it(cds):
    structures = [Atoms(symbols=["C", "H", "H", "H", "H", "H"], positions=cd) for cd in cds]
    # Setting up the ACSF descriptor
    acsf = ACSF(
        species=["C", "H"],
        rcut=6.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )
    acsf_water = acsf.create(structures)
    ryan_acsf = acsf_water.reshape((len(cds), len(acsf_water) // len(cds), -1))
    ryan_acsf2 = acsf_water.reshape((len(acsf_water) // len(cds), len(cds), -1))
    print('idk')
    return ryan_acsf


# def atm_atm_dists(cds, mat=False):
#     """
#     computes all pairwise atm-atm distances per walker
#     :param cds: nxmx3 cartesian coordinates
#     :return: (n, p), n = num_geoms, p = num_atm_atm_dists, 0,1 ; 0,2 ; 1,2 ; ...
#     """
#     from sklearn.metrics.pairwise import euclidean_distances
#     if mat:
#         dist_mats = np.zeros((len(cds), cds.shape[1], cds.shape[1]))
#         for geom_num, geom in enumerate(cds):
#             distz = euclidean_distances(geom, geom)
#             np.fill_diagonal(distz, 1.0)
#             dist_mats[geom_num] = distz
#         return dist_mats
#     else:
#         iu1 = np.triu_indices(len(cds[0]), k=1)
#         dist_vecs = np.zeros((len(cds), len(iu1[0])))
#         for geom_num, geom in enumerate(cds):
#             dist_mat = euclidean_distances(geom, geom)
#             dist_vecs[geom_num] = dist_mat[iu1]
#         return dist_vecs


def atm_atm_dists(cds, mat=False):
    ngeoms = cds.shape[0]
    natoms = cds.shape[1]
    idxs = np.transpose(np.triu_indices(natoms, 1))
    atoms_0 = cds[:, tuple(x[0] for x in idxs)]
    atoms_1 = cds[:, tuple(x[1] for x in idxs)]
    diffs = atoms_1 - atoms_0
    dists = np.linalg.norm(diffs, axis=2).T
    if mat:
        result = np.zeros((ngeoms, natoms, natoms))
        idxss = np.triu_indices_from(result[0], k=1)
        result[:, idxss[0], idxss[1]] = dists.T
        result += result.transpose(0,2,1)
        crap = np.broadcast_to(np.eye(natoms), result.shape)
        result += crap
        return result

def sort_coulomb(c_mat):
    indexlist = np.argsort(-1 * np.linalg.norm(c_mat, axis=1))
    # indexlist_pos = np.argsort(np.linalg.norm(c_mat, axis=1))
    sorted_c_mat = c_mat[np.arange(c_mat.shape[0])[:, None, None], indexlist[:, :, None], indexlist[:, None, :]]
    return sorted_c_mat


def coulomb_it(cds):
    zs = np.array([6, 1, 1, 1, 1, 1])

    # get 0.5 * z^0.4
    rest = np.ones((len(zs), len(zs)))
    np.fill_diagonal(rest, 0.5 * zs ** 0.4)

    # get zii^2/zij matrix
    zij = np.outer(zs, zs)

    # rij
    start = time.time()
    atm_atm_mat = atm_atm_dists(cds, mat=True)
    # print(f"Time 1: {time.time()-start}")
    # start = time.time()
    # atm_atm_mat2 = atm_atm_dists_2(cds, mat=True)
    # print(f"Time 2: {time.time() - start}")
    coulomb = zij * rest / atm_atm_mat

    # sort each according to norm of rows/columns
    coulomb_s = sort_coulomb(coulomb)
    return coulomb_s


def dscribe_coulomb(cds):
    structures = [Atoms(symbols=["C", "H", "H", "H", "H", "H"], positions=cd) for cd in cds]
    # Setting up the ACSF descriptor
    coul = CoulombMatrix(
        n_atoms_max=6,
        flatten=False,
        permutation='sorted_l2'
    )
    cm_ch5 = coul.create(structures)
    print('idk')
    return cm_ch5


def load_training(training_name):
    """If using deb_training_every argument, read the files with this."""
    with h5py.File(training_name, 'r') as f:
        cds = f['coords'][:]
        vs = f['pots'][:]
    return cds, Constants.convert(vs, "wavenumbers", to_AU=False)


def get_ml_data(ts):
    tot_x = []
    tot_v = []
    for time in ts:
        cds, vs = load_training(f"training_0_training_{time}ts.hdf5")
        print(cds.shape)
        tot_x.append(cds.squeeze())
        tot_v.append(vs)
    tot_x = np.concatenate(tot_x)
    tot_v = np.concatenate(tot_v)
    return tot_x, tot_v


def plot_training_data(cds, pots):
    plt.scatter(cds, pots)


tss = [500, 1000, 1500]
# tss = [500]
train_x, train_y = get_ml_data(tss)
# xyz_npy.write_xyz(train_x,
#                   'ch5.xyz',
#                   ["C", "H", "H", "H", "H", "H"],
#                   cmt=list(Constants.convert(train_y, 'wavenumbers', to_AU=True))
#                   )

train_x_permut = np.array([train_x[0], train_x[0]])
train_x_permut[1, [0, 1]] = train_x_permut[1, [1, 0]]
# start = time.time()
ryans_coulomb = coulomb_it(train_x_permut)
# ryans_coulomb = coulomb_it(train_x)
# print(f"Time Coulomb: {time.time()-start}")

# dscr_coul = dscribe_coulomb(train_x_permut) gives same results yay
print('hello')
