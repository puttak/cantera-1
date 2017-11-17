import time
import sys

import multiprocessing as mp
import pandas as pd
import numpy as np
import scipy.integrate

import matplotlib.pyplot as plt

import cantera as ct
print("Running Cantera version: {}".format(ct.__version__))


# gas = ct.Solution('gri30.xml')
gas = ct.Solution('./data/Boivin_newTherm.cti')
# Initial condition
P = ct.one_atm


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
    temp=ini[0]
    nH=ini[1]
    gas.TPX = temp, P, 'H2:'+str(nH)+',O2:1,N2:4'
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
        # Update the sample
        train_org.append(state_org)
        train_new.append(state_new)

        #if (abs(state_res.max()) < 1e-5 and solver.t > 0.0001):
        if (abs(state_res.max()) < 1e-3 and gas['H2'].Y < 0.006):
            break

    return train_org, train_new


def data_gen(ini):

    #ini_T = [1001, 1001, 1001, 1001]

    print("multiprocessing:", end='')
    t_start = time.time()
    p = mp.Pool(processes=mp.cpu_count())
    training_data = p.map(ignite, ini)
    p.close()

    org, new = zip(*training_data)
    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['temperature']
    # columnNames = columnNames+['pressure']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    t_end = time.time()
    tmp = t_end - t_start
    print(" %8.3f seconds" % tmp)
    return train_org,train_new


if __name__ == "__main__":
    ini_T = np.linspace(1001, 3001, 1)
    ini = [(temp, 2) for temp in ini_T]
    ini = ini+ [(temp, 1) for temp in ini_T]
    a,b=data_gen(ini)

    # ini_T = np.linspace(501, 3001, 100)
    # #ini_T = [1001, 1001, 1001, 1001]
    #
    # print("multiprocessing:", end='')
    # t_start = time.time()
    # p = mp.Pool(processes=mp.cpu_count())
    # training_data = p.map(ignite, ini_T)
    # p.close()
    #
    # org, new = zip(*training_data)
    # org = np.concatenate(org)
    # new = np.concatenate(new)
    #
    # columnNames = gas.species_names
    # columnNames = columnNames + ['temperature']
    # # columnNames = columnNames+['pressure']
    #
    # train_org = pd.DataFrame(data=org, columns=columnNames)
    # train_new = pd.DataFrame(data=new, columns=columnNames)
    #
    # t_end = time.time()
    # tmp = t_end - t_start
    # print(" %8.3f seconds" % tmp)




