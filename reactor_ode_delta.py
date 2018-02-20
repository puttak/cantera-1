import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

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
        # self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.set_unnormalized_mole_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        #dYdt = wdot * self.gas.molecular_weights / rho
        dYdt = wdot / rho

        return np.hstack((dTdt, dYdt))


def ignite(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    t_end = 1e-3
    dt = 1e-6
    for dt_ini in[1e-6]:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            gas = ct.Solution('./data/h2_sandiego.cti')
        if fuel == 'CH4':
            gas = ct.Solution('./data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
        # y0 = np.hstack((gas.T, gas.Y))
        x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        # solver.set_initial_value(y0, 0.0)
        solver.set_initial_value(x0, 0.0)

        while solver.successful() and solver.t < t_end:
            # state_org = np.hstack([gas[gas.species_names].Y, gas.T, dt])
            state_org = np.hstack([gas[gas.species_names].X, gas.T, dt])
            # state_org = np.hstack([gas[gas.species_names].X, gas.T,
            #                        np.dot(gas.partial_molar_enthalpies,gas.X)/gas.density, dt])
            if solver.t == 0:
                solver.integrate(solver.t + dt_ini)
            else:
                solver.integrate(solver.t + dt)
            # gas.TPY = solver.y[0], P, solver.y[1:]
            gas.TPX = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack([gas[gas.species_names].X, gas.T, dt])

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
            if ((res.max() < 1e-3 and (solver.t / dt) > 50)) or (gas['H2'].Y < 0.005 or gas['H2'].Y >0.995):
            # if res.max() < 1e-5:
                break

    return train_org, train_new


def ignite_random_x(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    dt = 1e-6
    t_end = 1e-3
    for dt_ini in[1e-6, 9e-7, 1.1e-6]:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            gas = ct.Solution('./data/h2_sandiego.cti')
        if fuel == 'CH4':
            gas = ct.Solution('./data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        rnd = np.random.RandomState(int(n_fuel))
        ini_x = rnd.rand(gas.X.size)
        ini_x /= ini_x.sum()
        gas.TPX = temp, P, ini_x
        # y0 = np.hstack((gas.T, gas.Y))
        x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        # solver.set_initial_value(y0, 0.0)
        solver.set_initial_value(x0, 0.0)

        while solver.successful() and solver.t < t_end:
            # state_org = np.hstack([gas[gas.species_names].Y, gas.T, dt])
            state_org = np.hstack([gas[gas.species_names].X, gas.T, dt])
            # state_org = np.hstack([gas[gas.species_names].X, gas.T,
            #                        np.dot(gas.partial_molar_enthalpies,gas.X)/gas.density, dt])

            if solver.t == 0:
                solver.integrate(solver.t + dt_ini)
            solver.integrate(solver.t + dt)
            # gas.TPY = solver.y[0], P, solver.y[1:]
            gas.TPX = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack([gas[gas.species_names].X, gas.T, dt])

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
            if ((res.max() < 1e-3 and (solver.t / dt) > 50)) or (gas['H2'].Y < 0.005 or gas['H2'].Y >0.995):
            # if res.max() < 1e-5:
                break

    return train_org, train_new


def ignite_post(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    t_end = 1e-3
    dt = 1e-6
    for dt_ini in[0]:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            gas = ct.Solution('./data/h2_sandiego.cti')
        if fuel == 'CH4':
            gas = ct.Solution('./data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
        # y0 = np.hstack((gas.T, gas.Y))
        x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        # solver.set_initial_value(y0, 0.0)
        solver.set_initial_value(x0, 0.0)

        while solver.successful() and solver.t < t_end:
            # state_org = np.hstack([gas[gas.species_names].Y, gas.T, dt])
            state_org = np.hstack([gas[gas.species_names].X, gas.T, dt])
            # state_org = np.hstack([gas[gas.species_names].X, gas.T,
            #                        np.dot(gas.partial_molar_enthalpies,gas.X)/gas.density, dt])
            solver.integrate(solver.t + dt)
            # gas.TPY = solver.y[0], P, solver.y[1:]
            gas.TPX = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack([gas[gas.species_names].X, gas.T, dt])

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
            if ((res.max() < 1e-3 and (solver.t / dt) > 50)) or (gas['H2'].Y < 0.005 or gas['H2'].Y >0.995):
            # if res.max() < 1e-5:
                break

    return train_org, train_new



def data_gen(ini_Tn, fuel):

    if fuel == 'H2':
        # gas = ct.Solution('./data/Boivin_newTherm.cti')
        gas = ct.Solution('./data/h2_sandiego.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')

    print("multiprocessing:", end='')
    t_start = time.time()
    p = mp.Pool(processes=mp.cpu_count())

    ini = [(x[0], x[1], fuel) for x in ini_Tn]
    # training_data = p.map(ignite_random_x, ini)
    training_data = p.map(ignite, ini)
    p.close()

    org, new = zip(*training_data)

    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['T']
    # columnNames = columnNames + ['dT']
    columnNames = columnNames+['dt']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    return train_org, train_new


if __name__ == "__main__":
    ini_T = np.linspace(1201, 1501, 1)
    ini = [(temp, 2) for temp in ini_T]
    ini = ini + [(temp, 10) for temp in ini_T]
    a, b = data_gen(ini, 'H2')