import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyvibdmc import *
from pyvibdmc.analysis import *


def load_data(fname):
    with h5py.File(fname, 'r') as f:
        cds = f['coords'][:]
        vs = f['pots'][:]
    return cds, Constants.convert(vs, "wavenumbers", to_AU=False)


# fig, axs = plt.subplots(3, 1)
# over time
# for dN, d in enumerate([1, 5, 10]):
#     for i in np.arange(500, 1000, 100):
#         cds, vs = load_data(f"training_water_dt{d}_1_training_{i}ts.hdf5")
#         axs[dN].hist(vs, bins=1000, range=(0, 70000), alpha=0.5)
#         axs[dN].set_xlim([0, 20000])
# fig.savefig("training_2.png", dpi=300, bbox_inches='tight')


cds, dw = SimInfo.get_wfn('wfns/training_water_dt5_1_wfn_800ts.hdf5')
cds, vs = load_data(f"training_water_dt5_1_training_800ts.hdf5")
analyzer = AnalyzeWfn(cds)
oh1 = analyzer.bond_length(0, 2)
oh2 = analyzer.bond_length(1, 2)
hh = analyzer.bond_length(0, 1)
hoh = np.degrees(analyzer.bond_angle(0, 2, 1))

bins_x, bins_y, amps = analyzer.projection_2d(oh1,
                                              vs,
                                              desc_weights=np.ones(len(cds)),
                                              range=[[1.2, 2.5], [0, 10000]],
                                              normalize=False)
Plotter.plt_hist2d(bins_x,
                   bins_y,
                   amps,
                   "ROH1(Angstroms)",
                   "Potential Energy",
                   save_name='roh_v.png')

bins_x, bins_y, amps = analyzer.projection_2d(oh1 + oh2,
                                              vs,
                                              desc_weights=np.ones(len(cds)),
                                              range=[[3, 4.5], [0, 10000]],
                                              normalize=False)
Plotter.plt_hist2d(bins_x,
                   bins_y,
                   amps,
                   "ROH1+ROH2 (Angstroms)",
                   "Potential Energy",
                   save_name='rohroh_v.png')

bins_x, bins_y, amps = analyzer.projection_2d(hoh,
                                              vs,
                                              desc_weights=np.ones(len(cds)),
                                              range=[[80, 140], [0, 10000]],
                                              normalize=False)
Plotter.plt_hist2d(bins_x,
                   bins_y,
                   amps,
                   "HOH (Degrees)",
                   "Potential Energy",
                   save_name='hoh_v.png')

bins_x, bins_y, amps = analyzer.projection_2d(oh1 + oh2 + hh,
                                              vs,
                                              desc_weights=np.ones(len(cds)),
                                              range=[[5, 9], [0, 10000]],
                                              normalize=False)
Plotter.plt_hist2d(bins_x,
                   bins_y,
                   amps,
                   "ROH1+ROH2+RHH (Angstroms)",
                   "Potential Energy",
                   save_name='all_v.png')
