import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import pandas as pd
import time
import multiprocessing as mp
import pandas as pd
import dask.dataframe as dd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import subprocess
import os

from reactor_ode_delta import ReactorOde

import cantera as ct


def one_step_pro(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    Y_ini = ini[1]
    dt = ini[2]
    fuel = ini[3]

    # dt_base = 1e-6
    # dt = dt_base * (0.9 + round(0.2 * np.random.random(), 2))

    if fuel == 'H2':
        # gas = ct.Solution('./data/Boivin_newTherm.cti')
        gas = ct.Solution('./data/h2_sandiego.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')
        # gas = ct.Solution('gri30.xml')
    P = ct.one_atm

    # gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
    for temp_org, Y_org, dt_org in zip(temp, Y_ini, dt):
        gas.TP = temp_org, P
        gas.X = Y_org

        # y0 = np.hstack((gas.T, gas.Y))
        x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        # solver.set_initial_value(y0, 0.0)
        solver.set_initial_value(x0, 0.0)

        state_org = np.hstack(
            [gas[gas.species_names].X, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
             gas.T, gas.density, gas.cp, dt_org, 0])

        solver.integrate(solver.t + dt_org)
        # gas.TPY = solver.y[0], P, solver.y[1:]
        gas.TPX = solver.y[0], P, solver.y[1:]

        # Extract the state of the reactor
        state_new = np.hstack(
            [gas[gas.species_names].X, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
             gas.T, gas.density, gas.cp, dt_org, 0])

        # Update the sample
        train_org.append(state_org)
        train_new.append(state_new)

    return train_org, train_new


def one_step_data_gen(df, fuel):
    gas = ct.Solution('./data/h2_sandiego.cti')

    # df=df[df['temperature[[K]']>900]
    # df = df.sample(70)
    Y_sp = df[gas.species_names]
    T = df['T']

    dt_base = 1e-6
    dt = dt_base * (0.9 + np.round(0.2 * np.random.random(df.shape[0]), 2))

    ini = [(a, b, c) for a, b, c in zip(np.array_split(T.values, mp.cpu_count()),
                                  np.array_split(Y_sp.values, mp.cpu_count()),
                                  np.array_split(dt, mp.cpu_count()))]



    print("multiprocessing:", end='')
    t_start = time.time()
    p = mp.Pool(processes=mp.cpu_count())

    ini = [(x[0], x[1], x[2], fuel) for x in ini]

    training_data = p.map(one_step_pro, ini)
    p.close()

    org, new = zip(*training_data)

    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['Hs']
    columnNames = columnNames + ['T']
    columnNames = columnNames + ['Rho']
    columnNames = columnNames + ['cp']
    columnNames = columnNames + ['dt']
    columnNames = columnNames + ['f']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    return train_org, train_new


if __name__ == '__main__':

    new_test = pickle.load(open('data/test_griddata.p', 'rb'))

    gas = ct.Solution('./data/h2_sandiego.cti')


    new_test['N2'] = 1 - new_test[gas.species_names[:-1]].sum(1)
    df_x_input_org, df_y_target_org = one_step_data_gen(new_test, 'H2')
    pickle.dump((df_x_input_org, df_y_target_org), open('data/x_y_org_new.p', 'wb'))




