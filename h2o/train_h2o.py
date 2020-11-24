import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *
from sklearn.neural_network import MLPRegressor


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
        cds, vs = load_training(f"training_water_dt1_1_training_{time}ts.hdf5")
        print(cds.shape)
        tot_x.append(cds.squeeze())
        tot_v.append(vs)
    tot_x = np.concatenate(tot_x)
    tot_v = np.concatenate(tot_v)
    return tot_x, tot_v


def plot_training_data(cds, pots):
    plt.scatter(cds, pots)

def clean_training_data(train_x, train_y,diffmat):
    threshold = 0.001
    # death_matels = np.zeros(diffmat.shape)
    final_x = np.zeros(train_x.shape)
    for row_num,row in enumerate(diffmat):
        #calculate threshold
        diffz = row[row_num+1:]
        death_mark = np.abs(diffz) < threshold
        final_x += np.concatenate((np.zeros(row_num+1),death_mark))
        # death_matels[row_num, np.where(death_mark)[0]] = 1
    np.delete(train_x, final_x)
    np.delete(train_y, final_x)
    print('hi')
    train_x

def difference_matrix(train_x):
    x = np.reshape(train_x, (len(train_x), 1))
    return x - x.transpose()

def internals_h2O(cds):
    analyzer = AnalyzeWfn(cds)
    bl1 = analyzer.bond_length(0,2)
    bl2 = analyzer.bond_length(1,2)
    theta = analyzer.bond_angle(0,2,1)
    return np.array((bl1,bl2,theta)).T

tss = [100, 200, 300, 400, 500]
tss = [100, 200]
train_x, train_y = get_ml_data(tss)
train_x = internals_h2O(train_x)
# diffmat = difference_matrix(train_x)
# clean_training_data(train_x, train_y,diffmat)

val_set = [800]
val_x, val_y = get_ml_data(val_set)
val_x = internals_h2O(val_x)
# max_train_x = np.amax(train_x)
# max_train_y = np.amax(train_y)
# norm_train_x = train_x / max_train_x
# norm_train_y = train_y / max_train_y

# plot_training_data(train_x, train_y)
model = MLPRegressor(activation='relu',
                     solver='adam',
                     max_iter=200,
                     hidden_layer_sizes=(3, 3),
                     # alpha=0.1,
                     )
# train_x = train_x.reshape(-1, 1)
# val_x = val_x.reshape(-1, 1)
mfit = model.fit(train_x, train_y)
yhat = model.predict(train_x)
loss = np.square(yhat - train_y).mean()
print(loss)
# plt.scatter(val_x, yhat, c='r')
# plt.scatter(yhat,val_y)
plt.show()