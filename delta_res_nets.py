import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import pandas as pd
from deltaNets import combustionML
from boost_test import test_data, tot, create_data

if __name__ == '__main__':
    # %%
    create_data()
    # load training
    df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    columns = df_x.columns
    # train_features = columns.drop(['f', 'dt'])
    train_features = columns.drop(['f','N2'])

    # initial conditions
    n_H2 = sorted(list(map(float, set(df_x['f']))))
    n_H2 = np.asarray(n_H2).reshape(-1, 1)

    df_x = df_x[train_features]
    df_y = df_y[train_features]

    # drop df_x == 0
    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]

    # target
    res_dict = {'y/x': df_y / df_x,
                'y': df_y,
                'y-x': df_y - df_x}
    res = res_dict['y/x']

    # species = ['O']
    # species = train_features
    species = train_features.drop(['dt'])

    target = pd.DataFrame(res[species], columns=species)
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
    # nn_std = combustionML(df_x, target, {'x': 'log_std', 'y': 'log_std'})
    # nn_std = combustionML(df_x, target, {'x': 'std2', 'y': 'std2'})
    nn_std = combustionML(df_x, target, {'x': 'log', 'y': 'log'})

    r2 = nn_std.run([400, 2, 0., 8000])
    nn_std.plt_loss()

    # %%
    # test
    batch_predict = 1024*256
    ensemble_mode = True
    # ensemble_mode = False

    # post_species = {'T'}
    # post_species = species
    post_species = species.drop(['cp', 'Hs', 'T', 'Rho'])

    ini_T = 1501
    for sp in post_species.intersection(species):
        for n in [13]:
            input, test = test_data(ini_T, n, columns)
            # input['C'] = tot(input, 'C')
            # input['tot:O'] = tot(input, 'O')
            # input['tot:H'] = tot(input, 'H')
            input = input.drop(['N2'], axis=1)
            # input = input.drop(['N2', 'dt'], axis=1)
            if ensemble_mode is True:
                pred = pd.DataFrame(nn_std.inference_ensemble(input,batch_size=batch_predict), columns=target.columns)
            else:
                pred = pd.DataFrame(nn_std.inference(input), columns=target.columns)

            model_pred = pred
            pred = pred * input

            test_target = test / input
            # test_target = test

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test[sp])
            axarr[0].plot(pred[sp], 'rd', ms=2)
            # axarr[0].set_title(str(n) + '_' + sp)

            axarr[1].plot((test[sp] - pred[sp]) / test[sp], 'k')
            axarr[1].set_ylim(-0.005, 0.005)
            # axarr[1].set_title(str(n) + '_' + sp)
            f.suptitle(str(n) + '_' + sp)

            ax2 = axarr[1].twinx()
            ax2.plot(test_target[sp], 'bd', ms=2)
            ax2.plot(model_pred[sp], 'rd', ms=2)
            ax2.set_ylim(0.8, 1.2)
            plt.savefig('fig/' + str(n) + '_' + sp)
            plt.show()
#%%
    for sp in post_species.intersection(species):
        for n in [13]:
            input, test = test_data(ini_T, n, columns)
            # input['C'] = tot(input, 'C')
            # input['tot:O'] = tot(input, 'O')
            # input['tot:H'] = tot(input, 'H')
            input = input.drop(['N2'], axis=1)
            test = test.drop(['N2'], axis=1)
            # input = input.drop(['N2', 'dt'], axis=1)
            # test = test.drop(['N2', 'dt'], axis=1)
            init = 50
            input_model = input.values[init].reshape(1, -1)
            test_model = test.values[init].reshape(1, -1)
            pred = test_model
            # for i in range(input.shape[0] - (init + 1)):
            #     pred_model = nn_std.inference(input_model)
            #     pred = np.vstack((pred, input_model * pred_model))
            #     input_model = input_model * pred_model
            for i in range(input.shape[0] - (init + 1)):

                if ensemble_mode is True:
                    pred_model = nn_std.inference_ensemble(input_model,batch_size=batch_predict)
                else:
                    pred_model = nn_std.inference(input_model)

                input_model[0][:-1] = input_model[0][:-1] * pred_model
                input_model[0][-1] = 1e-6
                pred = np.vstack((pred, input_model))

            pred = pd.DataFrame(pred, columns=train_features)

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test[sp].values[init:],'bd', ms=2)
            axarr[0].plot(pred[sp], 'rd', ms=2)

            axarr[1].plot(abs(test[sp].values[init:] - pred[sp]) / test[sp].values[init:], 'k')
            # axarr[1].set_ylim(-0.005, 0.005)
            f.suptitle('Intigration: '+str(n) + '_' + sp)
            plt.savefig('fig/acc_' + str(n) + '_' + sp)
            plt.show()

    #%%

    # a = nn_std.inference_ensemble(df_x, batch_size=batch_predict)
    test_all = df_x
    test_all = df_x.astype('float32').values
    t_start = time.time()
    a = nn_std.inference_ensemble(test_all, batch_size=1024*256)
    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))
