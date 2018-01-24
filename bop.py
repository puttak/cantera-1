import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_lowercase

from sklearn import model_selection, metrics
from sklearn.preprocessing import normalize

# import cntk
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_floatx('float32')
print("precision: " + K.floatx())

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from res_block import res_block
from reactor_ode_p import data_gen, data_scaling, data_inverse, ignite

import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))
# T = np.linspace(1501, 2001, 15)
# n = np.linspace(1, 0., 15)
# XX, YY = np.meshgrid(T, n)
# ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)
#
# df_x_input, df_y_target = data_gen(ini, 'CH4')

# x_input, y_target = shuffle(x_input, y_target)
# y_target = y_target.drop('temperature', axis=1)

# x_input, norm_x, std_x = data_scaling(df_x_input)
# y_target, norm_y, std_y = data_scaling(df_y_target)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_target,
#                                                                     test_size=0.1,
#                                                                     random_state=42)


# x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x_input, df_y_target,
#                                                                     test_size=0.1,
#                                                                     random_state=42)
# x_train, norm_x, std_x = data_scaling(x_train)
# y_train, norm_y, std_y = data_scaling(y_train)
# x_test, _, _ = data_scaling(x_test,norm_x,std_x)
# y_test, _, _ = data_scaling(y_test,norm_y,std_y)
def dl_react(nn_s, nn_l, temp, n_fuel, ini=None):
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
    # print(state_org)
    while t < t_end:
        train_org.append(state_org)

        # inference
        print(state_org[0,1])
        if state_org[0,1]>nn_s.df_x_input['H'].max():
            state_new = nn_l.inference(state_org)
        else:
            state_new = nn_s.inference(state_org)

        #state_new_norm = state_new/state_new[0,:-1].sum()
        #state_new_norm[0,-1]=state_new[0,-1]
        #print(state_new[0,:-1].sum(),state_new_norm[0,:-1].sum())

        # print(state_new)
        # print(state_new[0, :-1].sum())

        train_new.append(state_new)
        state_res = state_new - state_org
        print(state_new[0,1])

        # Update the sample
        state_org = state_new
        t = t + dt
        if abs(state_res.max() / state_org.max()) < 1e-4 and (t / dt) > 100:
            break
        # if state_org[0, 1] > 1e-8:
        #     break

    train_org = np.concatenate(train_org, axis=0)
    train_org = pd.DataFrame(data=train_org, columns=nn_l.df_x_input.columns)
    train_new = np.concatenate(train_new, axis=0)
    train_new = pd.DataFrame(data=train_new, columns=nn_l.df_y_target.columns)
    return train_org, train_new


def cmp_plot(nn_s, nn_l, n_fuel, sp, st_step):
    for temp in [1001, 1101, 1201]:
        start = st_step

        # ode integration
        ode_o, ode_n = ignite((temp, n_fuel, 'H2'))
        ode_o = np.delete(ode_o, 8, 1)
        ode_n = np.delete(ode_n, 8, 1)
        ode_o = pd.DataFrame(data=np.asarray(ode_o),
                             columns=nn_s.df_x_input.columns)
        ode_n = pd.DataFrame(data=np.asarray(ode_n),
                             columns=nn_s.df_y_target.columns)

        cmpr = nn_s.inference(ode_o)
        cmpr = pd.DataFrame(data=cmpr,
                            columns=nn_s.df_x_input.columns)

        dl_o, dl_n = dl_react(nn_s, nn_l, temp, n_fuel, ini=ode_o.values[start].reshape(1, -1))

        plt.figure()

        ode_show = ode_n[sp][start:].values
        cmpr_show = cmpr[sp][start:].values
        dl_show = dl_n[sp]

        plt.semilogy(ode_show, 'kd', label='ode', ms=1)
        plt.semilogy(dl_show, 'b-', label='dl')
        plt.semilogy(cmpr_show, 'r:', label='cmpr')
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)
    return dl_o,dl_n

def cut_plot(nn_s,nn_l , n_fuel, sp, st_step):
    for temp in [1001, 1101, 1201]:
        start = st_step

        # ode integration
        ode_o, ode_n = ignite((temp, n_fuel, 'H2'))
        ode_o = np.delete(ode_o, 8, 1)
        ode_n = np.delete(ode_n, 8, 1)
        ode_o = pd.DataFrame(data=np.asarray(ode_o),
                             columns=nn_l.df_x_input.columns)
        ode_n = pd.DataFrame(data=np.asarray(ode_n),
                             columns=nn_l.df_y_target.columns)

        # cmpr = test.inference(ode_o)
        # cmpr = pd.DataFrame(data=cmpr,
        #                     columns=test.df_x_input.columns)

        dl_o, dl_n = dl_react(nn_s, nn_l, temp, n_fuel, ini=ode_o.values[start].reshape(1, -1))

        plt.figure()

        ode_show = ode_n[sp][start:].values
        # cmpr_show = cmpr[sp][start:].values
        dl_show = dl_n[sp]

        plt.semilogy(ode_show, 'kd', label='ode', ms=1)
        plt.semilogy(dl_show, 'b-', label='dl')
        # plt.semilogy(cmpr_show, 'r:', label='cmpr')
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)
    return dl_o,dl_n


class combustionML(object):
    # def __init__(self, df_x_input, df_y_target, x_train, x_test, y_train, y_test):
    def __init__(self, df_x_input, df_y_target):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x_input, df_y_target,
                                                                            test_size=0.1,
                                                                            random_state=42)
        self.x_train, self.norm_x, self.std_x = data_scaling(x_train)
        self.y_train, self.norm_y, self.std_y = data_scaling(y_train)
        x_test, _, _ = data_scaling(x_test, self.norm_x, self.std_x)
        # y_test, _, _ = data_scaling(y_test,norm_y,std_y)

        self.df_x_input = df_x_input
        self.df_y_target = df_y_target
        self.x_test = pd.DataFrame(data=x_test, columns=df_x_input.columns)
        self.y_test = pd.DataFrame(data=y_test, columns=df_y_target.columns)

        self.model = None
        self.history = None
        self.callbacks_list = None
        self.vsplit = None
        self.predict = None

    def composeResnetModel(self, n_neurons=200, blocks=2, drop1=0.1, loss='mse', optimizer='adam', batch_norm=False):
        ######################
        print('set up ANN')
        # ANN parameters
        dim_input = self.x_train.shape[1]
        dim_label = self.y_train.shape[1]

        # This returns a tensor
        inputs = Input(shape=(dim_input,), dtype='float32')
        print(inputs.dtype)
        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(n_neurons, name='1_base')(inputs)
        # x = BatchNormalization(axis=-1, name='1_base_bn')(x)
        x = Activation('relu')(x)

        # less then 2 res_block, there will be variance
        for b in range(blocks):
            x = res_block(x, n_neurons, stage=1, block=ascii_lowercase[b], d1=drop1, bn=batch_norm)

        predictions = Dense(dim_label, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=predictions)

        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # checkpoint (save the best model based validate loss)
        filepath = "./tmp/weights.best.cntk.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=10)
        self.callbacks_list = [checkpoint]

    def fitModel(self, batch_size=1024, epochs=400, vsplit=0.3):
        # fit the model
        self.vsplit = vsplit
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=vsplit,
            verbose=2,
            callbacks=self.callbacks_list,
            shuffle=True)

    def prediction(self):
        self.model.load_weights("./tmp/weights.best.cntk.hdf5")

        predict = self.model.predict(self.x_test.values)
        predict = data_inverse(predict, self.norm_y, self.std_y)
        self.predict = pd.DataFrame(data=predict, columns=self.df_y_target.columns)

        R2_score = -abs(metrics.r2_score(predict, self.y_test))
        print(R2_score)
        return R2_score

    def inference(self, x):
        # self.model.load_weights("./tmp/weights.best.cntk.hdf5")
        # print(' new')
        tmp, _, _ = data_scaling(x, self.norm_x, self.std_x)
        predict = self.model.predict(tmp)
        # inverse for out put
        out = data_inverse(predict, self.norm_y, self.std_y)
        # eliminate negative values
        out[out < 0] = 0
        # normalized total mass fraction to 1
        # out_y = out[:, :-1]
        # out_y = normalize(out_y, axis=1, norm='l1')
        # out_norm = np.concatenate((out_y, out[:, -1].reshape(-1, 1)), axis=1)

        out_y = out[:, :-1]
        out_y = normalize(out_y, axis=1, norm='l1') * (np.asarray(x)[:, :-1].sum(1).reshape(-1,1))
        out_norm = np.concatenate((out_y, out[:, -1].reshape(-1, 1)), axis=1)

        return out

    def acc_plt(self, sp):
        plt.figure()
        plt.plot(self.y_test[sp], self.predict[sp], 'kd', ms=1)
        plt.axis('tight')
        plt.axis('equal')

        # plt.axis([train_new[sp].min(), train_new[sp].max(), train_new[sp].min(), train_new[sp].max()], 'tight')
        r2 = round(metrics.r2_score(self.y_test[sp], self.predict[sp]), 6)
        plt.title(sp + ' : r2 = ' + str(r2))

    # loss
    def plt_loss(self):
        # fig = plt.figure()
        plt.semilogy(self.history.history['loss'])
        if self.vsplit:
            plt.semilogy(self.history.history['val_loss'])
        plt.title('mae')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

    def run(self, hyper):
        print(hyper)

        self.composeResnetModel(n_neurons=hyper[0], blocks=hyper[1], drop1=hyper[2])
        self.fitModel(epochs=1000, batch_size=1024 * 8 )

        return self.prediction()


if __name__ == "__main__":
    T = np.linspace(1001, 2001, 20)
    n = np.linspace(8, 0., 40)
    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

    df_x_input, df_y_target = data_gen(ini, 'H2')

    df_x_input = df_x_input.drop('N2', axis=1)
    df_y_target = df_y_target.drop('N2', axis=1)

    df_cut_s = df_y_target['H'] < 1e-8
    df_cut_l = df_y_target['H'] >= 1e-8

    df_x_input_l = df_x_input[df_cut_l]
    df_y_target_l = df_y_target[df_cut_l]
    df_x_input_l = df_x_input_l.reset_index(drop=True)
    df_y_target_l = df_y_target_l.reset_index(drop=True)

    df_x_input_s = df_x_input[df_cut_s]
    df_y_target_s = df_y_target[df_cut_s]
    df_x_input_s = df_x_input_s.reset_index(drop=True)
    df_y_target_s = df_y_target_s.reset_index(drop=True)

    nn_l = combustionML(df_x_input_l, df_y_target_l)
    nn_s = combustionML(df_x_input_s, df_y_target_s)

    bop = False
    if bop:
        from skopt import gp_minimize

        res = gp_minimize(nn_l.run,  # the function to minimize
                          [(100, 200, 300, 400, 500, 600),
                           (2, 5),
                           (0., 0.1, 0.2, 0.3, 0.4, 0.5)],  # the bounds on each dimension of x

                          acq_func="EI",  # the acquisition function
                          n_calls=15,  # the number of evaluations of f
                          n_random_starts=5,  # the number of random initialization points
                          random_state=123)  # the random seed

        from skopt.plots import plot_convergence

        plot_convergence(res);
        print(res.x, res.fun)

    nn_s.run([400, 2, 0.])
    nn_l.run([400, 2, 0.])
    # test.composeResnetModel(400,2,0.)
    # test.prediction()
    sp = 'H'
    #plt.plot(test.x_test[sp], test.y_test[sp], 'kd', ms=1)
    nn_s.acc_plt(sp)

