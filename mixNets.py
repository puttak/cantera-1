import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_lowercase

from sklearn import model_selection, metrics
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_floatx('float32')
print("precision: " + K.floatx())
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from res_block import res_block
from reactor_ode_p import data_gen, ignite
from tmp import data_scaling, data_inverse

import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))

import pickle


def dl_react(nns, temp, n_fuel, swt, ini=None):
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
        state_std = nns[0].inference(state_org)
        state_log = nns[1].inference(state_org)

        state_new = state_log
        acc, _, _ = data_scaling(state_new, nns[1].scaler_case, nns[1].norm_y, nns[1].std_y)
        for i in range(9):
            # if state_log[0, i] > swt:
            if acc[0, i] > swt and state_new[0, i] > 1e-4:
                state_new[0, i] = state_std[0, i]

        state_new[0, 0] = state_org[0, 0] + state_org[0, 1] + 1.0 / 17 * state_org[0, 3] \
                          + 2.0 / 18 * state_org[0, 5] + 1.0 / 33 * state_org[0, 6] + 2.0 / 34 * state_org[0, 7] \
                          - state_new[0, 1] - 1.0 / 17 * state_new[0, 3] \
                          - 2.0 / 18 * state_new[0, 5] - 1.0 / 33 * state_new[0, 6] - 2.0 / 34 * state_new[0, 7]
        state_new[0,0] = max(state_new[0,0],0)
        # print(state_new[0,0])

        train_new.append(state_new)
        state_res = state_new - state_org
        res = abs(state_res[state_org != 0] / state_org[state_org != 0])

        # Update the sample
        state_org = state_new
        t = t + dt
        # if abs(state_res.max() / state_org.max()) < 1e-4 and (t / dt) > 100:
        # if res.max() < 1e-5 and (t / dt) > 100:
        #     break
        if state_org[0, :-1].sum() > 1.5:
            break

    train_org = np.concatenate(train_org, axis=0)
    train_org = pd.DataFrame(data=train_org, columns=nns[0].df_x_input.columns)
    train_new = np.concatenate(train_new, axis=0)
    train_new = pd.DataFrame(data=train_new, columns=nns[0].df_y_target.columns)
    return train_org, train_new


def cmp_plot(nns, n_fuel, sp, st_step, swt):
    for temp in [1001]:
        # for temp in [1001, 1101, 1201]:
        start = st_step

        # ode integration
        ode_o, ode_n = ignite((temp, n_fuel, 'H2'))
        ode_o = np.delete(ode_o, 8, 1)
        ode_n = np.delete(ode_n, 8, 1)
        ode_o = pd.DataFrame(data=np.asarray(ode_o),
                             columns=nns[0].df_x_input.columns)
        ode_n = pd.DataFrame(data=np.asarray(ode_n),
                             columns=nns[0].df_y_target.columns)
        cmpr = []
        for input in ode_o.values:

            input = input.reshape(1, -1)

            # inference
            # print(input)
            state_std = nns[0].inference(input)
            # state_std[state_std<1e-4] = 0

            state_log = nns[1].inference(input)
            # state_log[state_log>=1e-4] = 0

            state_new = state_log
            #            print(state_new)
            acc, _, _ = data_scaling(input, nns[1].scaler_case, nns[1].norm_y, nns[1].std_y)
            for i in range(9):
                # print(i)
                # print(state_log)
                # if state_log[0, i] > swt:
                if acc[0, i] > swt:
                    # print(acc[0,i])
                    state_new[0, i] = state_std[0, i]
                    # if abs((state_log[0, i] - state_std[0, i]) / state_log[0, i]) > 1e-2:
                    #     state_new[0, i] = state_std[0, i]+state_log[0,i]

            cmpr.append(state_new)
        cmpr = np.concatenate(cmpr, axis=0)
        cmpr = pd.DataFrame(data=cmpr,
                            columns=nns[0].df_x_input.columns)

        ode_show = ode_n[sp][start:].values
        cmpr_show = cmpr[sp][start:].values

        # plt.figure()
        #
        # plt.semilogy(ode_show, 'kd', label='ode', ms=1)
        # plt.semilogy(cmpr_show, 'r:', label='cmpr_s')
        # plt.legend()
        # plt.title('ini_t = ' + str(temp) + ': ' + sp)

        if swt * (1 - swt) == 0:
            a, _, _ = data_scaling(cmpr, nns[swt].scaler_case, nns[swt].norm_y, nns[swt].std_y)
            a = pd.DataFrame(data=a,
                             columns=nns[swt].df_x_input.columns)
            plt.figure()
            plt.plot(a[sp][start:].values)

        plt.figure()
        plt.plot(abs(cmpr_show - ode_show) / ode_show, 'kd', ms=1)

        plt.figure()
        plt.plot(ode_show, 'kd', label='ode', ms=1)
        plt.plot(cmpr_show, 'rd', label='cmpr_s', ms=1)
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)

    return cmpr, ode_o, ode_n


def cut_plot(nns, n_fuel, sp, st_step, swt):
    # for temp in [1001, 1101, 1201]:
    for temp in [1001]:
        start = st_step

        # ode integration
        ode_o, ode_n = ignite((temp, n_fuel, 'H2'))
        ode_o = np.delete(ode_o, 8, 1)
        ode_n = np.delete(ode_n, 8, 1)
        ode_o = pd.DataFrame(data=np.asarray(ode_o),
                             columns=nns[0].df_x_input.columns)
        ode_n = pd.DataFrame(data=np.asarray(ode_n),
                             columns=nns[0].df_y_target.columns)

        dl_o, dl_n = dl_react(nns, temp, n_fuel, swt, ini=ode_o.values[start].reshape(1, -1))

        plt.figure()

        ode_show = ode_n[sp][start:].values
        dl_show = dl_n[sp][:ode_show.size]

        plt.semilogy(ode_show, 'kd', label='ode', ms=1)
        plt.semilogy(dl_show, 'bd', label='dl', ms=1)
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)

        # plt.figure()
        # plt.plot(abs(dl_show[ode_show != 0] - ode_show[ode_show != 0]) / ode_show[ode_show != 0], 'kd', ms=1)

    return dl_n


class classScaler(object):
    def __init__(self):
        self.norm = None
        self.std = None

    def fit_transform(self, input):
        self.norm = MinMaxScaler()
        self.std = StandardScaler()
        out = self.std.fit_transform(input)
        out = self.norm.fit_transform(out)
        return out

    def transform(self, input):
        out = self.std.transform(input)
        out = self.norm.transform(out)

        return out


class cluster(object):
    def __init__(self, data, T):
        self.T_ = T
        self.labels_ = np.asarray((data['T'] > self.T_).astype(int))

    def predict(self, input):
        out = (input[:, -1] > self.T_).astype(int)
        return out


class combustionML(object):
    # def __init__(self, df_x_input, df_y_target, x_train, x_test, y_train, y_test):
    def __init__(self, df_x_input, df_y_target, scaler_case):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x_input, df_y_target,
                                                                            test_size=0.1,
                                                                            random_state=42)
        self.x_train, self.norm_x, self.std_x = data_scaling(x_train, scaler_case)
        self.y_train, self.norm_y, self.std_y = data_scaling(y_train, scaler_case)
        x_test, _, _ = data_scaling(x_test, scaler_case, self.norm_x, self.std_x)

        self.scaler_case = scaler_case
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
        floatx = 'float32'
        #K.set_floatx('floatx')
        # ANN parameters
        dim_input = self.x_train.shape[1]
        dim_label = self.y_train.shape[1]

        # This returns a tensor
        inputs = Input(shape=(dim_input,), dtype=floatx)

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

    # fit the model
    def fitModel(self, batch_size=1024, epochs=400, vsplit=0.3):

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
        predict = data_inverse(predict, self.scaler_case, self.norm_y, self.std_y)
        self.predict = pd.DataFrame(data=predict, columns=self.df_y_target.columns)

        R2_score = -abs(metrics.r2_score(predict, self.y_test))
        print(R2_score)
        return R2_score

    def inference(self, x):

        tmp, _, _ = data_scaling(x, self.scaler_case, self.norm_x, self.std_x)
        predict = self.model.predict(tmp)
        # inverse for out put
        out = data_inverse(predict, self.scaler_case, self.norm_y, self.std_y)
        # eliminate negative values
        out[out < 0] = 0
        # normalized total mass fraction to 1
        # out_y = out[:, :-1]
        # out_y = normalize(out_y, axis=1, norm='l1')
        # out_norm = np.concatenate((out_y, out[:, -1].reshape(-1, 1)), axis=1)

        # out_y = out[:, :-1]
        # out_y = normalize(out_y, axis=1, norm='l1') * (np.asarray(x)[:, :-1].sum(1).reshape(-1, 1))
        # out_norm = np.concatenate((out_y, out[:, -1].reshape(-1, 1)), axis=1)

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
        self.fitModel(epochs=700, batch_size=1024 * 8)

        return self.prediction()


if __name__ == "__main__":
    T = np.linspace(801, 2001, 20)
    n = np.linspace(8, 0., 20)
    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

    # generate data
    df_x_input, df_y_target = data_gen(ini, 'H2')
    # drop inert N2
    df_x_input = df_x_input.drop('N2', axis=1)
    df_y_target = df_y_target.drop('N2', axis=1)

    # create two nets
    nns = []
    r2s = []

    nn_std = combustionML(df_x_input, df_y_target, 'std')
    r2 = nn_std.run([400, 4, 0.1])
    r2s.append(r2)
    nns.append(nn_std)

    nn_log = combustionML(df_x_input[df_y_target['H']<1e-6], df_y_target[df_y_target['H']<1e-6], 'log')
    r2 = nn_log.run([400, 2, 0.1])
    r2s.append(r2)
    nns.append(nn_log)

    # dl_react(nns, class_scaler, kmeans, 1001, 2, df_x_input_l.values[0].reshape(1,-1))
    # cut_plot(nns, class_scaler, kmeans, 2, 'H', 0)

    a, b_o, b_n = cmp_plot(nns, 4, 'O2', 0, 1)
    c = abs(b_n[b_o != 0] - b_o[b_o != 0]) / b_o[b_o != 0]
