import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from reactor_ode_p import data_gen, ignite, data_scaling, data_inverse
from bop import combustionML

import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))

T = np.linspace(1001, 3101, 40)
n = np.linspace(4, 0., 40)
XX, YY = np.meshgrid(T, n)
ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

df_x_input, df_y_target = data_gen(ini, 'H2')
test = combustionML(df_x_input, df_y_target)
test.composeResnetModel(400, 4, 0.1)
test.prediction()


def dl_react(temp, n_fuel, ini=None):
    gas = ct.Solution('./data/Boivin_newTherm.cti')
    # gas = ct.Solution('./data/grimech12.cti')

    fuel = 'H2'
    gas.X = fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
    gas.TP = temp, ct.one_atm

    # dl model
    t_end = 1e-3
    dt = 1e-6
    t = 0

    train_org = []
    train_new = []
    state_org = np.hstack([gas[gas.species_names].Y, gas.T]).reshape(1, -1)
    if ini.any() != None:
        state_org = ini
    while t < t_end:
        train_org.append(state_org)

        # inference
        state_new = test.inference(state_org)
        train_new.append(state_new)
        state_res = state_new - state_org

        # Update the sample
        state_org = state_new
        t = t + dt
        if abs(state_res.max() / state_org.max()) < 1e-4 and (t / dt) > 50:
            break

    train_org = np.concatenate(train_org, axis=0)
    train_org = pd.DataFrame(data=train_org, columns=test.df_x_input.columns)
    train_new = np.concatenate(train_new, axis=0)
    train_new = pd.DataFrame(data=train_new, columns=test.df_y_target.columns)
    return train_org, train_new


def plot(n_fuel, sp, st_step):
    for temp in [1001, 1501, 2001, 2501]:
        start = st_step

        # ode integration
        ode_o, ode_n = ignite((temp, n_fuel, 'H2'))
        ode_o = pd.DataFrame(data=np.asarray(ode_o),
                             columns=test.df_x_input.columns)
        ode_n = pd.DataFrame(data=np.asarray(ode_n),
                             columns=test.df_y_target.columns)

        cmpr = test.inference(ode_o)
        cmpr = pd.DataFrame(data=cmpr,
                            columns=test.df_x_input.columns)

        dl_o, dl_n = dl_react(temp, n_fuel, ini=ode_o.values[start].reshape(1, -1))

        plt.figure()

        ode_show = ode_n[sp][start:].values
        cmpr_show = cmpr[sp][start:].values
        dl_show = dl_n[sp]

        plt.plot(ode_show, 'kd', label='ode', ms=1)
        plt.plot(dl_show, 'b-', label='dl')
        plt.plot(cmpr_show, 'r:', label='cmpr')
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)
    return ode_o
