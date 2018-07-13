import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

from reactor_ode_delta import ignite_post, data_gen_f
import pandas as pd
from deltaNets import combustionML
from boost_test import test_data, tot, create_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from dataScaling import LogScaler, AtanScaler, NoScaler

if __name__ == '__main__':
    # %%
    # create_data()
    # load training
    df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    columns = df_x.columns
    train_features = columns.drop(['f', 'dt'])

    n_H2 = set(df_x['f'])
    n_H2 = np.asarray(sorted(list(map(float, n_H2)))).reshape(-1, 1)

    df_x = df_x[train_features]
    df_y = df_y[train_features]

    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]

    # target
    res_dict = {'y/x': df_y / df_x,
                'y': df_y,
                'y-x': df_y - df_x}
    res = res_dict['y/x']

    scaler_dict = {'log': LogScaler(),
                   'no': NoScaler(),
                   'mmx': MinMaxScaler(),
                   'mabs': MaxAbsScaler(),
                   'std': StandardScaler(),
                   'atan': AtanScaler()}
    scaler = scaler_dict['no']

    # species = ['O']
    # species = ['O2', 'H2', 'OH', 'O', 'H2O']
    # species = train_features.drop(['dt'])
    species = train_features

    target = pd.DataFrame(scaler.fit_transform(res[species]), columns=species)
    outlier = 1.2

    idx = (target < outlier).all(1)
    out_ratio = idx.sum() / target.shape[0]

    target = target.loc[idx]
    df_x = df_x.loc[idx]

    # add new features
    # df_x['C'] = tot(df_x, 'C')
    # df_x['tot:O'] = tot(df_x, 'O')
    # df_x['tot:H'] = tot(df_x, 'H')

    # %%
    # model formulate
    # nn_std = combustionML(df_x, target, 'std2')
    # nn_std = combustionML(df_x, target, {'x': 'log_std', 'y': 'log_std'})
    nn_std = combustionML(df_x, target, {'x': 'log', 'y': 'log'})
    # nn_std = combustionML(df_x, target, {'x': 'std2', 'y': 'std2'})
    r2 = nn_std.run([200, 2, 0.])
    nn_std.plt_loss()

    # %%
    # test
    # post_species = {'O','O2'}
    post_species = species
    ini_T = 1501
    for sp in post_species.intersection(species):
        for n in [3]:
            input, test = test_data(ini_T, n, columns)
            # input['C'] = tot(input, 'C')
            # input['tot:O'] = tot(input, 'O')
            # input['tot:H'] = tot(input, 'H')
            input = input.drop(['dt'], axis=1)
            pred = pd.DataFrame(scaler.inverse_transform(nn_std.inference(input)), columns=target.columns)
            model_pred = pred
            pred = pred * input

            test_target = test / input
            # test_target = test

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test[sp])
            axarr[0].plot(pred[sp], 'rd', ms=2)
            # axarr[0].set_title(sp + ':' + str(r2_score(test.values.reshape(-1,1), sp_pred)))

            # axarr[1].plot((test[sp] - input[sp]) / test[sp], 'r')
            axarr[1].plot((test[sp] - pred[sp]) / test[sp], 'k')
            axarr[1].set_ylim(-0.005, 0.005)
            axarr[1].set_title(str(n) + '_' + sp)

            ax2 = axarr[1].twinx()
            ax2.plot(test_target[sp], 'bd', ms=2)
            ax2.plot(model_pred[sp], 'rd', ms=2)
            ax2.set_ylim(0.8, 1.2)
            plt.savefig('fig/' + str(n) + '_' + sp)
            plt.show()

    # %%
    for sp in post_species.intersection(species):
        for n in [13]:
            input, test = test_data(ini_T, n, columns)
            # input['C'] = tot(input, 'C')
            # input['tot:O'] = tot(input, 'O')
            # input['tot:H'] = tot(input, 'H')
            input = input.drop(['dt'], axis=1)
            init = 100
            input_model = input.values[init].reshape(1, -1)
            pred = input_model
            for i in range(input.shape[0]-(init+1)):
                pred_model = nn_std.inference(input_model)
                pred = np.vstack((pred, input_model * pred_model))
                input_model = input_model * pred_model
            pred = pd.DataFrame(pred, columns=target.columns)

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test[sp].values[init:],'bd', ms=2)
            axarr[0].plot(pred[sp], 'rd', ms=2)
            axarr[0].set_title(str(n) + ':' + sp)

            # axarr[1].plot(pred[sp], 'rd', ms=2)

            axarr[1].plot(abs(test[sp].values[init:] - pred[sp]) / test[sp].values[init:], 'k')
            # axarr[1].set_ylim(-0.005, 0.005)
            # axarr[1].set_title(str(n) + '_' + sp)
            plt.savefig('fig/acc_' + str(n) + '_' + sp)
            plt.show()
