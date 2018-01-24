import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_lowercase

from sklearn import model_selection, metrics
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans

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


def dl_react(nns, class_scaler,kmeans, temp, n_fuel, ini=None):
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

        # select neural network
        # print(np.reshape(ini, (1, -1)))
        # b=np.reshape(ini, (1, -1))
        # print(b.shape)
        tmp = class_scaler.transform(ini)
        a = kmeans.predict(tmp)
        for i in a:
            nn=nns[i]

        # if state_org[0, 1] > swt:
        #     nn = nn_l
        #     print('l')
        #     # break
        # else:
        #     nn = nn_s

        # inference
        state_new = nn.inference(state_org)

        # state_new_norm = state_new/state_new[0,:-1].sum()
        # state_new_norm[0,-1]=state_new[0,-1]
        # print(state_new[0,:-1].sum(),state_new_norm[0,:-1].sum())

        # print(state_new)
        # print(state_new[0, :-1].sum())

        train_new.append(state_new)
        state_res = state_new - state_org
        res = abs(state_res[state_org != 0] / state_org[state_org != 0])

        # Update the sample
        state_org = state_new
        t = t + dt
        # if abs(state_res.max() / state_org.max()) < 1e-4 and (t / dt) > 100:
        if res.max() < 1e-4 and (t / dt) > 100:
            break
        if state_org[0, 1] > 1e-1:
            break

    train_org = np.concatenate(train_org, axis=0)
    train_org = pd.DataFrame(data=train_org, columns=nn.df_x_input.columns)
    train_new = np.concatenate(train_new, axis=0)
    train_new = pd.DataFrame(data=train_new, columns=nn.df_y_target.columns)
    return train_org, train_new


def cmp_plot(nn_s, nn_l, swt, n_fuel, sp, st_step):
    for temp in [1001]:
        # for temp in [1001, 1101, 1201]:
        start = st_step

        # ode integration
        ode_o, ode_n = ignite((temp, n_fuel, 'H2'))
        ode_o = np.delete(ode_o, 8, 1)
        ode_n = np.delete(ode_n, 8, 1)
        ode_o = pd.DataFrame(data=np.asarray(ode_o),
                             columns=nn_s.df_x_input.columns)
        ode_n = pd.DataFrame(data=np.asarray(ode_n),
                             columns=nn_s.df_y_target.columns)

        cmpr_s = nn_s.inference(ode_o)
        cmpr_s = pd.DataFrame(data=cmpr_s,
                              columns=nn_s.df_x_input.columns)

        cmpr_l = nn_l.inference(ode_o)
        cmpr_l = pd.DataFrame(data=cmpr_l,
                              columns=nn_l.df_x_input.columns)

        dl_o, dl_n = dl_react(nn_s, nn_l, swt, temp, n_fuel, ini=ode_o.values[start].reshape(1, -1))

        plt.figure()

        ode_show = ode_n[sp][start:].values
        cmpr_s_show = cmpr_s[sp][start:].values
        cmpr_l_show = cmpr_l[sp][start:].values
        dl_show = dl_n[sp]

        plt.semilogy(ode_show, 'kd', label='ode', ms=1)
        plt.semilogy(dl_show, 'b-', label='dl')
        plt.semilogy(cmpr_s_show, 'r:', label='cmpr_s')
        plt.semilogy(cmpr_l_show, 'r-', label='cmpr_l')
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)

        # plt.figure()
        # plt.plot(ode_show, 'kd', label='ode', ms=1)
        # plt.plot(dl_show, 'b-', label='dl')
        # plt.plot(cmpr_s_show, 'r:', label='cmpr_s')
        # plt.plot(cmpr_l_show, 'r-', label='cmpr_l')
        # plt.legend()
        # plt.title('ini_t = ' + str(temp) + ': ' + sp)

    return dl_o, dl_n


def cut_plot(nns, class_scaler,kmeans, n_fuel, sp, st_step):
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

        dl_o, dl_n = dl_react(nns, class_scaler, kmeans, temp, n_fuel, ini=ode_o.values[start].reshape(1, -1))

        plt.figure()

        ode_show = ode_n[sp][start:].values
        dl_show = dl_n[sp]

        plt.semilogy(ode_show, 'kd', label='ode', ms=1)
        plt.semilogy(dl_show, 'b-', label='dl')
        plt.legend()
        plt.title('ini_t = ' + str(temp) + ': ' + sp)



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
        out_y = normalize(out_y, axis=1, norm='l1') * (np.asarray(x)[:, :-1].sum(1).reshape(-1, 1))
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
        self.fitModel(epochs=400, batch_size=1024 * 8)

        return self.prediction()


if __name__ == "__main__":
    T = np.linspace(1001, 1201, 10)
    n = np.linspace(8, 0., 10)
    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

    df_x_input, df_y_target = data_gen(ini, 'H2')

    df_x_input = df_x_input.drop('N2', axis=1)
    df_y_target = df_y_target.drop('N2', axis=1)

    # cut_scale = 1e-9
    # df_cut_s = df_y_target['O']+df_y_target['H'] < cut_scale
    # df_cut_l = df_y_target['O'] <= 1e-5
    #
    # df_x_input_l = df_x_input[df_cut_l*df_y_target['O']>1e-10]
    # df_y_target_l = df_y_target[df_cut_l*df_y_target['O']>1e-10]
    # df_x_input_l = df_x_input_l.reset_index(drop=True)
    # df_y_target_l = df_y_target_l.reset_index(drop=True)
    #
    #
    # df_x_input_s = df_x_input[df_cut_s]
    # df_y_target_s = df_y_target[df_cut_s]
    # df_x_input_s = df_x_input_s.reset_index(drop=True)
    # df_y_target_s = df_y_target_s.reset_index(drop=True)

    tot_clusters = 8
    class_scaler = MinMaxScaler()
    data = class_scaler.fit_transform(df_x_input)
    kmeans = KMeans(n_clusters=tot_clusters).fit(data)

    nns = []
    for i in range(tot_clusters):
        clt = kmeans.labels_ == i

        df_x_input_l = df_x_input[clt]
        df_y_target_l = df_y_target[clt]
        df_x_input_l = df_x_input_l.reset_index(drop=True)
        df_y_target_l = df_y_target_l.reset_index(drop=True)

        nn = combustionML(df_x_input_l, df_y_target_l)

        nn.run([400, 2, 0.])

        sp = 'H'
        nn.acc_plt(sp)
        nns.append(nn)

    dl_react(nns,class_scaler,kmeans,1001,2,df_x_input_l.values[0])

    # bop = False
    # if bop:
    #     from skopt import gp_minimize
    #
    #     res = gp_minimize(nn_l.run,  # the function to minimize
    #                       [(100, 200, 300, 400, 500, 600),
    #                        (2, 5),
    #                        (0., 0.1, 0.2, 0.3, 0.4, 0.5)],  # the bounds on each dimension of x
    #
    #                       acq_func="EI",  # the acquisition function
    #                       n_calls=15,  # the number of evaluations of f
    #                       n_random_starts=5,  # the number of random initialization points
    #                       random_state=123)  # the random seed
    #
    #     from skopt.plots import plot_convergence
    #
    #     plot_convergence(res);
    #     print(res.x, res.fun)
