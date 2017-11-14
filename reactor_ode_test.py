"""
Solve a constant pressure ignition problem where the governing equations are
implemented in Python.

This demonstrates an approach for solving problems where Cantera's reactor
network model cannot be configured to describe the system in question. Here,
Cantera is used for evaluating thermodynamic properties and kinetic rates while
an external ODE solver is used to integrate the resulting equations. In this
case, the SciPy wrapper for VODE is used, which uses the same variable-order BDF
methods as the Sundials CVODES solver used by Cantera.
"""

import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))

import numpy as np
import pandas as pd
import scipy.integrate

import matplotlib.pyplot as plt


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


# gas = ct.Solution('gri30.xml')
gas = ct.Solution('./data/Boivin_newTherm.cti')

# Initial condition
P = ct.one_atm
# gas.TPX = 1001, P, 'H2:2,O2:1,N2:4'
# y0 = np.hstack((gas.T, gas.Y))

# now compile a list of all variables for which we will store data
columnNames = gas.species_names
columnNames = columnNames+['temperature']
# columnNames = columnNames+['pressure']

# use the above list to create a DataFrame
train_new = pd.DataFrame(columns=columnNames)
train_org = pd.DataFrame(columns=columnNames)
train_res = pd.DataFrame(columns=columnNames)

# Set up objects representing the ODE and the solver
# ode = ReactorOde(gas)
# solver = scipy.integrate.ode(ode)
# solver.set_integrator('vode', method='bdf', with_jacobian=True)
# solver.set_initial_value(y0, 0.0)

# Integrate the equations, keeping T(t) and Y(k,t)
t_end = 1e-3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = 1e-6
i = 0
input_val = []
target_val = []
ini_T =np.linspace(1301,1501,1)
for temp in ini_T:
    gas.TPX = temp, P, 'H2:2,O2:1,N2:4'
    y0 = np.hstack((gas.T, gas.Y))
    ode = ReactorOde(gas)
    solver = scipy.integrate.ode(ode)
    solver.set_integrator('vode', method='bdf', with_jacobian=True)
    solver.set_initial_value(y0, 0.0)
    while solver.successful() and solver.t < t_end:
        state_old = np.hstack([gas[gas.species_names].Y, gas.T])
        # state_old = np.hstack([gas[gas.species_names].Y])
        solver.integrate(solver.t + dt)
        gas.TPY = solver.y[0], P, solver.y[1:]
        states.append(gas.state, t=solver.t)
        # Extract the state of the reactor
        state_new = np.hstack([gas[gas.species_names].Y, gas.T])
        # state_new = np.hstack([gas[gas.species_names].Y])
        state_res = state_new - state_old
        #if(abs(state_res.max())<1e5):
#            break
        print(abs(state_res).max())
        # Update the dataframe
        # train_new.loc[solver.t] = state_new
        # train_org.loc[solver.t] = state_old
        # train_res.loc[solver.t] = state_res
        train_new.loc[i] = state_new
        train_org.loc[i] = state_old
        train_res.loc[i] = state_res
        i = i + 1
        input_val.append(state_old)
        target_val.append(state_new)
        if(abs(state_res.max())<1e-5 and solver.t >0.0001):
            break

train_org = train_org.loc[:, (train_org != 0).any(axis=0)]
train_new = train_new.loc[:, (train_new != 0).any(axis=0)]
train_res = train_res.loc[:, (train_res != 0).any(axis=0)]

if __name__ == "__main__":
    # L1 = plt.plot(states.t, states.T, color='r', label='T', lw=2)
    L1 = plt.plot(states.T, color='r', label='T', lw=2)
    plt.xlabel('time (s)')
    plt.ylabel('Temperature (K)')
    plt.twinx()
    # L2 = plt.plot(states.t, states('OH').Y, label='OH', lw=2)
    L2 = plt.plot(states('OH').Y, label='OH', lw=2)
    plt.ylabel('Mass Fraction')
    plt.legend(L1 + L2, [line.get_label() for line in L1 + L2], loc='lower right')

    plt.figure()
    plt.semilogx(train_new.index, train_new['H2'], '-o')

    plt.show()
