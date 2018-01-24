import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import scipy.integrate

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))


class ReactorOde(object):
    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


def ignite(ini):
    t_end = 1e-3
    dt = 1e-6
    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    if fuel == 'H2':
        gas = ct.Solution('./data/Boivin_newTherm.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')
        # gas = ct.Solution('gri30.xml')
    P = ct.one_atm

    gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
    y0 = np.hstack((gas.T, gas.Y))
    ode = ReactorOde(gas)
    solver = scipy.integrate.ode(ode)
    solver.set_integrator('vode', method='bdf', with_jacobian=True)
    solver.set_initial_value(y0, 0.0)
    train_org = []
    train_new = []
    while solver.successful() and solver.t < t_end:
        state_org = np.hstack([gas[gas.species_names].Y, gas.T])
        # state_org = np.hstack([gas[gas.species_names].Y])
        solver.integrate(solver.t + dt)
        gas.TPY = solver.y[0], P, solver.y[1:]
        # Extract the state of the reactor
        state_new = np.hstack([gas[gas.species_names].Y, gas.T])
        # state_new = np.hstack([gas[gas.species_names].Y])
        state_res = state_new - state_org
        res = abs(state_res[state_org!=0]/state_org[state_org!=0])
        # res[res==np.inf]=0
        # res = np.nan_to_num(res)
        # res=res[res!=0]
        # print(res.max())

        # Update the sample
        train_org.append(state_org)
        train_new.append(state_new)

        # if (abs(state_res.max() / state_org.max()) < 1e-5 and (solver.t / dt) > 200):
        if (res.max() < 1e-4 and (solver.t / dt) > 100):
            break

    return train_org, train_new


def data_gen(ini_Tn, fuel):
    # ini_T = [1001, 1001, 1001, 1001]

    if fuel == 'H2':
        gas = ct.Solution('./data/Boivin_newTherm.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')

    print("multiprocessing:", end='')
    t_start = time.time()
    p = mp.Pool(processes=mp.cpu_count())

    ini = [(x[0], x[1], fuel) for x in ini_Tn]
    training_data = p.map(ignite, ini)
    p.close()

    org, new = zip(*training_data)
    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['T']
    # columnNames = columnNames+['P']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    return train_org, train_new


def data_scaling(input, norm=None, std=None):
    # if not norm:
    #     # print(1)
    #     norm = MinMaxScaler()
    #     std = StandardScaler()
    #     out = std.fit_transform(input)
    #     out = 2 * norm.fit_transform(out) - 1
    # else:
    #     # print(2)
    #     out = std.transform(input)
    #     out = 2 * norm.transform(out) - 1

    if not norm:
        # print(1)
        norm = MinMaxScaler()
        std = StandardScaler()
        out = std.fit_transform(input)
        out = norm.fit_transform(out)
    else:
        # print(2)
        out = std.transform(input)
        out = norm.transform(out)

    # if not norm or not std:
    #     norm = MinMaxScaler()
    #     std = StandardScaler()
    #     out = norm.fit_transform(input)
    # else:
    #     out = norm.transform(input)

    # if not norm or not std:
    #     norm = MinMaxScaler()
    #     std = StandardScaler()
    #     out = std.fit_transform(input)
    # else:
    #     out = std.transform(input)

    return out, norm, std


def data_inverse(input, norm, std):
    # out = norm.inverse_transform(0.5 * (input + 1))
    # out = std.inverse_transform(out)

    out = norm.inverse_transform(input)
    out = std.inverse_transform(out)

    # print('min max norm')
    # out = norm.inverse_transform(input)

    # print('std norm')
    # out = std.inverse_transform(input)

    return np.double(out)


if __name__ == "__main__":
    ini_T = np.linspace(1001, 3001, 1)
    ini = [(temp, 2) for temp in ini_T]
    ini = ini + [(temp, 1) for temp in ini_T]
    a, b = data_gen(ini, 'H2')
