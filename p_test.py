import time
import sys
from joblib import Parallel, delayed
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


def ignite(temp):
    t_end = 1e-3
    dt = 1e-6
    gas.TPX = temp, P, 'H2:2,O2:1,N2:4'
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

        if (abs(state_res.max()) < 1e-5 and solver.t > 0.0001):
            break

    return train_org, train_new


# A function that can be called to do work:
def work(arg):
    # cpu_id = int(mp.current_process().name.split('-')[1])
    # print(cpu_id)

    print("Function receives the arguments as a list:", arg)
    # Split the list to individual variables:
    i, j = arg
    out_org = []
    out_new = []
    for itr in range(3):
        state_org = np.hstack([i, i])
        state_new = np.hstack([j, j])
        out_org.append(state_org)
        out_new.append(state_new)
    # df.loc[cpu_id]=state
    # All this work function does is wait 1 second...
    time.sleep(1)
    # ... and prints a string containing the inputs:
    print("%s_%s" % (i, j))

    return np.asarray(out_org), np.asarray(out_new)


if __name__ == "__main__":
    # arg_instances = [(1, 1), (1, 2), (1, 3), (1, 4)]
    # Anything returned by work() can be stored:
    # results = Parallel(n_jobs=4, verbose=1, backend="threading")(map(delayed(work), arg_instances))
    # results = Parallel(n_jobs=4, verbose=1)(delayed(work)(arg_instances))
    # print(results)
    # training_data = Parallel(n_jobs=2, verbose=1, backend="threading")(map(delayed(ignite), ini_T))
    # org,new=zip(*training_data)

    ini_T = np.linspace(501, 2001, 100)
    # ini_T = [1001, 1001, 1001, 1001]
    print("multiprocessing:", end='')
    tstart = time.time()
    p = mp.Pool(processes=mp.cpu_count())
    training_data = p.map(ignite, ini_T)
    org, new = zip(*training_data)
    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['temperature']
    # columnNames = columnNames+['pressure']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    tend = time.time()
    tmp = tend - tstart
    print(" %8.3f seconds" % tmp)

    print("serial:         ", end='')
    sys.stdout.flush()
    tstart = time.time()
    serial_solutions = [ignite(temp) for temp in ini_T]
    org_s, new_s = zip(*serial_solutions)
    org_s = np.concatenate(org_s)
    new_s = np.concatenate(new_s)

    columnNames = gas.species_names
    columnNames = columnNames + ['temperature']
    # columnNames = columnNames+['pressure']

    train_org_s = pd.DataFrame(data=org_s, columns=columnNames)
    train_new_s = pd.DataFrame(data=new_s, columns=columnNames)
    tend = time.time()
    tserial = tend - tstart
    print(" %8.3f seconds" % tserial)


