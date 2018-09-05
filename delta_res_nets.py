import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import glob

import pandas as pd
from deltaNets import combustionML
from boost_test import test_data, tot, create_data

if __name__ == '__main__':
    # %%
    # create_data()

    # clean start
    files = glob.glob('./tmp/history/*.hdf5') \
            + glob.glob('./tmp/*.hdf5')
    for file in files:
        os.remove(file)

    # load training
    df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))

    df_x_new, df_y_new = pickle.load(open('data/x_y_org_new.p', 'rb'))
    df_x = df_x.append(df_x_new, ignore_index=True)
    df_y = df_y.append(df_y_new, ignore_index=True)

    # df_x['OH_inv'] = 1 - df_x['OH']
    # df_y['OH_inv'] = 1 - df_y['OH']

    columns = df_x.columns
    # train_features = columns.drop(['f', 'dt'])
    train_features = columns.drop(['f', 'N2'])

    # initial conditions
    n_H2 = sorted(list(map(float, set(df_x['f']))))
    n_H2 = np.asarray(n_H2).reshape(-1, 1)

    df_x = df_x[train_features]
    df_y = df_y[train_features]

    # drop df_x == 0
    indx = (df_x != 0).all(1)
    # indx = df_x['H2O2']>1e-8
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]

    # target
    res_dict = {'y/x': df_y / df_x,
                'y': df_y,
                'y-x': df_y - df_x,
                'log(y)': np.log(df_y + 1e-20),
                'log(y)/log(x)': np.log(df_y + 1e-20) / np.log(df_x)}

    # case = 'log(y)/log(x)'
    case = 'y/x'
    # case = 'y'
    # case = 'y-x'
    res = res_dict[case]
    # res = res_dict['y/x']
    # res = res_dict['y']
    # res = res_dict['log(y)']

    # species = ['O']
    # species = train_features
    species = train_features.drop(['dt'])

    target = pd.DataFrame(res[species], columns=species)
    outlier = 1.5

    idx = (target < outlier).all(1)

    out_ratio = idx.sum() / target.shape[0]

    target_train = target.loc[idx]
    df_x = df_x.loc[idx]

    # add new features
    # df_x['C'] = tot(df_x, 'C')
    # df_x['tot:O'] = tot(df_x, 'O')
    # df_x['tot:H'] = tot(df_x, 'H')

    # %%
    # model formulate
    # nn_std = combustionML(df_x, target, {'x': 'log_std', 'y': 'log_std'})
    # nn_std = combustionML(df_x, target, {'x': 'std2', 'y': 'std2'})
    nn_std = combustionML(df_x, target_train, {'x': 'std', 'y': 'std'})
    # nn_std.ensemble_num = 5
    r2 = nn_std.run([200, 2, 0., 50])

    nn_std.plt_loss()

# %%
    # test interpolation
    batch_predict = 1024 * 256
    ensemble_mode = True
    # ensemble_mode = False

    # post_species = {'T'}
    # post_species = species
    post_species = species.drop(['cp', 'Hs', 'T', 'Rho'])
    import cantera as ct

    gas = ct.Solution('./data/h2_sandiego.cti')
    weights_dict = dict(zip(gas.species_names, gas.molecular_weights))
    weights = [weights_dict[k] for k in post_species]

    ini_T = 1801
    for sp in post_species.intersection(species):
        # for sp in ['OH_inv']:
        for n in [3]:
            input, test = test_data(ini_T, n, columns)
            # input['C'] = tot(input, 'C')
            # input['tot:O'] = tot(input, 'O')
            # input['tot:H'] = tot(input, 'H')
            # input['OH_inv'] = 1 - input['OH']
            # test['OH_inv'] = 1 - test['OH']

            input = input.drop(['N2'], axis=1)
            # input = input.drop(['N2', 'dt'], axis=1)
            if ensemble_mode is True:
                pred = pd.DataFrame(nn_std.inference_ensemble(input, batch_size=batch_predict), columns=target.columns)
            else:
                pred = pd.DataFrame(nn_std.inference(input), columns=target.columns)

            model_pred = pred

            if case == 'y/x':
                pred = pred * input
                test_target = test / input
            if case == 'log(y)/log(x)':
                pred = np.exp(np.log(input + 1e-20) * pred)
                test_target = np.log(test) / np.log(input + 1e-20)
            if case == 'y':
                pred = pred
                test_target = test
            if case == 'y-x':
                pred = input+pred
                test_target = test - input

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

    # %%
    # integration
    for sp in post_species.intersection(species):
        for n in [3]:
            input, test = test_data(ini_T, n, columns)
            # input['C'] = tot(input, 'C')
            # input['tot:O'] = tot(input, 'O')
            # input['tot:H'] = tot(input, 'H')
            input = input.drop(['N2'], axis=1)
            test = test.drop(['N2'], axis=1)
            # input['OH_inv'] = 1 - input['OH']
            # test['OH_inv'] = 1 - test['OH']
            init = 40
            input_model = input.values[init].reshape(1, -1)
            test_model = test.values[init].reshape(1, -1)
            pred_acc = test_model
            # for i in range(input.shape[0] - (init + 1)):
            #     pred_model = nn_std.inference(input_model)
            #     pred = np.vstack((pred, input_model * pred_model))
            #     input_model = input_model * pred_model
            for i in range(input.shape[0] - (init + 1)):

                if ensemble_mode is True:
                    pred_model = nn_std.inference_ensemble(input_model, batch_size=batch_predict)
                else:
                    pred_model = nn_std.inference(input_model)

                org_tot = sum(input_model[0][:8] * weights)

                if case == 'y/x':
                    input_model[0][:-1] = input_model[0][:-1] * pred_model
                    # a = list(range(12))
                    # a.append(13)
                    # input_model[0][a] = input_model[0][a] * pred_model
                if case == 'log(y)/log(x)':
                    input_model[0][:-1] = np.exp(np.log(input_model[0][:-1] + 1e-20) * pred_model)
                if case == 'y':
                    input_model[0][:-1] = pred_model
                if case == 'y-x':
                    input_model[0][:-1] = pred_model + input_model[0][:-1]
                input_model[0][-2] = 1e-6

                updated_tot = sum(input_model[0][:8] * weights)
                # input_model[0][:-1] = input_model[0][:-1] * org_tot / updated_tot

                pred_acc = np.vstack((pred_acc, input_model))

            pred_acc = pd.DataFrame(pred_acc, columns=train_features)

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test[sp].values[init:], 'bd', ms=2)
            axarr[0].plot(pred_acc[sp], 'rd', ms=2)
            # axarr[0].set_ylim(0, test[sp].max())

            axarr[1].plot(abs(test[sp].values[init:] - pred_acc[sp]) / test[sp].values[init:], 'k')
            # axarr[1].set_ylim(-0.005, 0.005)
            f.suptitle('Intigration: ' + str(n) + '_' + sp)
            plt.savefig('fig/acc_' + str(n) + '_' + sp)
            plt.show()

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].plot(test['OH'].values[init:], 'kd', ms=2)
    # axarr[0].plot(pred_acc['OH'], 'bd', ms=2)
    # axarr[0].plot(1 - pred_acc['OH_inv'], 'rd', ms=2)
    # # axarr[0].set_ylim(0, test[sp].max())
    #
    # axarr[1].plot(abs(test['OH'].values[init:] - (1 - pred_acc['OH_inv'])) / test['OH'].values[init:], 'r')
    # axarr[1].plot(abs(test['OH'].values[init:] - pred_acc['OH']) / test['OH'].values[init:], 'b')
    # # axarr[1].set_ylim(-0.005, 0.005)
    # f.suptitle('Intigration: ' + str(n) + '_' + 'OH')
    # plt.savefig('fig/acc_OH_inv_test_' + str(n) + 'OH')
    # plt.show()

    # %%
    # test_all = df_x
    # test_all = df_x.astype('float32').values
    # t_start = time.time()
    # a = nn_std.inference_ensemble(test_all, batch_size=1024 * 32)
    # t_end = time.time()
    # print(" %8.3f seconds" % (t_end - t_start))

    # post_species = post_species.drop('OH_inv')
    # pred_sum = (pred[post_species] * weights).sum(1)
    # pred_acc_sum = (pred_acc[post_species] * weights).sum(1)
    # test_sum = (test[post_species] * weights).sum(1)

    # plt.plot(pred_sum)
    # plt.plot(pred_acc_sum)
    # plt.plot(test_sum)
    # plt.show()
