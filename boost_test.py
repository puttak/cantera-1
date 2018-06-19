# %%
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

from reactor_ode_delta import ignite_post, data_gen_f
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from dataScaling import LogScaler, AtanScaler, NoScaler

# create data
def create_data():
    T = np.random.rand(20) * 1000 + 1001
    n_s = np.random.rand(15) * 7.6 + 0.1
    n_l = np.random.rand(30) * 40

    n = np.concatenate((n_s, n_l))
    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

    df_x_input_org, df_y_target_org = data_gen_f(ini, 'H2')
    pickle.dump((df_x_input_org, df_y_target_org), open('data/x_y_org.p', 'wb'))


def test_data(temp, n_fuel, columns):
    # temp = 1501
    # n_fuel = 4
    ode_o, ode_n = ignite_post((temp, n_fuel, 'H2'))
    ode_o = np.asarray(ode_o)
    ode_n = np.asarray(ode_n)
    ode_o = ode_o[ode_o[:, -1] == 1e-6]
    ode_n = ode_n[ode_n[:, -1] == 1e-6]
    ode_o = np.append(ode_o, n_fuel * np.ones((ode_o.shape[0], 1)), axis=1)
    ode_n = np.append(ode_n, n_fuel * np.ones((ode_n.shape[0], 1)), axis=1)
    ode_o = pd.DataFrame(data=ode_o,
                         columns=columns)
    ode_n = pd.DataFrame(data=ode_n,
                         columns=columns)

    # ode_o = ode_o.drop('N2', axis=1)
    ode_o = ode_o.drop('f', axis=1)
    ode_n = ode_n.drop('N2', axis=1)
    return ode_o, ode_n


def sp_plot_gpu(species, models, do, do_1):
    test = do_1[species]
    dtest = xgb.DMatrix(do)
    # sp_pred = models[species].predict(dtest)
    sp_pred = np.exp(models[species].predict(dtest))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(test)
    plt.plot(sp_pred, 'rd', ms=2)
    plt.title(species + ':' + str(r2_score(test, sp_pred)))

    plt.subplot(1, 2, 2)
    plt.plot((test - sp_pred) / test)
    plt.ylim(-0.1, 0.1)
    plt.show()
    return sp_pred


def sp_plot_gpu_mask(n, species, models, scalers, do, do_1, mask):
    plt.figure()
    for state in {'b:', 'u:'}:
        mask = ~mask
        if mask.any():
            test = do_1[mask][species]
            dtest = xgb.DMatrix(do)

            # sp_pred = np.exp(models[state + species].predict(dtest))
            sp_pred = scalers[state + species].inverse_transform(
                models[state + species].predict(dtest).reshape(-1, 1))
            # sp_pred = sp_pred[mask]

            sp_pred = sp_pred[mask] * do[mask][species].values.reshape(-1, 1) \
                      + do[mask][species].values.reshape(-1, 1)

            plt.subplot(1, 2, 1)
            plt.plot(test.index, test)
            plt.plot(test.index, sp_pred, 'rd', ms=2)
            plt.title(species + ':' + str(r2_score(test.values.reshape(-1,1), sp_pred)))

            plt.subplot(1, 2, 2)
            plt.plot((test.values.reshape(-1, 1) - sp_pred.reshape(-1,1)) / test.values.reshape(-1, 1))
            plt.ylim(-0.1, 0.1)
            plt.title(str(n) + '_' + species)

    plt.show()


if __name__ == '__main__':
    # generate data
    # create_data()

    # load training
    df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    columns = df_x.columns
    # df_x = df_x.drop('N2', axis=1)
    f = set(df_x['f'])
    f = np.asarray(sorted(list(map(float, f)))).reshape(-1, 1)
    df_x = df_x.drop('f', axis=1)
    df_y = df_y.drop('N2', axis=1)

    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]
#%%
    # target study
    res = (df_y - df_x) / df_x[df_x != 0]
    # res = (df_y - df_x)
    # res = df_y

    # data distribution analysis
    qt = res.drop(res.columns.intersection(['N2','dt','f']),axis=1)
    qt.hist()
    plt.show()
    qt_std = StandardScaler().fit_transform(qt)

    max = qt_std.max(0)
    min = qt_std.min(0)

    schew = sum(qt_std>1)/qt_std.shape[0] + \
            sum(qt_std < -1) / qt_std.shape[0]

    #%%
    df_x['C'] = df_x['H2'] + df_x['H2O']

    df_x['tot:O'] = 2 * df_x['O2'] + df_x['OH'] + df_x['O'] + df_x['H2O'] \
                    + 2 * df_x['HO2'] + 2 * df_x['H2O2']

    df_x['tot:H'] = 2 * df_x['H2'] + df_x['H'] + df_x['OH'] + 2 * df_x['H2O'] \
                    + df_x['HO2'] + 2 * df_x['H2O2']

    # %%
    # mask = df_x['O2'] < 0.01
    # mask_train = (df_x['O2'] < 0.01) | (df_x['H2'] < 0.01)
    mask_train = df_x['HO2'] < 1
    bstReg_gpu = {}
    scalers = {}
    species = ['T']

    # species = ['H2','OH','O2','O','H2O']
    # species = columns.drop(['dt','N2','f'])

    for state in {'b:', 'u:'}:
        mask_train = ~mask_train

        if mask_train.any():
            df_x_masked = df_x.loc[mask_train]
            df_y_masked = res.loc[mask_train]
            # df_y_masked[df_y_masked>0.2]=np.nan
            X_train, X_test, y_train, y_test = train_test_split(df_x_masked, df_y_masked,
                                                                test_size=.1, random_state=42)

            for sp in species:
                # scaler = LogScaler()
                # scaler = AtanScaler()
                # scaler = NoScaler()
                # scaler = MinMaxScaler()
                # scaler = MaxAbsScaler()
                scaler = StandardScaler()

                outlier = 100
                # target_train = np.log(y_train[sp])
                target_train = scaler.fit_transform(y_train[sp].values.reshape(-1, 1))
                # dtrain = xgb.DMatrix(X_train, label=target_train)
                dtrain = xgb.DMatrix(X_train[target_train < outlier], label=target_train[target_train < outlier])

                # target_test = np.log(y_test[sp])
                target_test = scaler.transform(y_test[sp].values.reshape(-1, 1))
                # dtest = xgb.DMatrix(X_test, label=target_test)
                dtest = xgb.DMatrix(X_test[target_test < outlier], label=target_test[target_test < outlier])
                param = {
                    'max_depth': 10,
                    'eta': 0.3,
                    'silent': 1,
                    'eval_metric': 'mae',
                    'predictor': 'gpu_predictor',
                    'objective': 'gpu:reg:linear'
                }

                num_round = 500
                bst = xgb.train(param, dtrain, num_round,
                                evals=[(dtest, 'test')], early_stopping_rounds=20)

                bstReg_gpu[state + sp] = bst
                scalers[state + sp] = scaler
                # print(sp + ':', r2_score(np.exp(bst.predict(dtest)), target_test))
                print(sp + ':', r2_score(np.exp(bst.predict(dtest)), target_test[target_test < outlier]))

    xgb.plot_importance(bst)
    plt.show()

    std=StandardScaler().fit_transform(target_train)
    # load test
    # %%

    for sp_test in species:
        # for n in [.5, 1.4, 2.6, 5, 10, 13, 25]:
        for n in [1, 25]:
            ode_o, ode_n = test_data(1501, n, columns)
            ode_o['C'] = ode_o['H2'] + ode_o['H2O']
            ode_o['tot:O'] = 2 * ode_o['O2'] + ode_o['OH'] + ode_o['O'] + ode_o['H2O'] \
                             + 2 * ode_o['HO2'] + 2 * ode_o['H2O2']
            ode_o['tot:H'] = 2 * ode_o['H2'] + ode_o['H'] + ode_o['OH'] + 2 * ode_o['H2O'] \
                             + ode_o['HO2'] + 2 * ode_o['H2O2']
            # mask_pred = (ode_o['H2'] < 0.01) | (ode_o['O2'] < 0.01)
            mask_pred = ode_o['HO2'] < 1
            # mask_pred = ~mask_pred
            sp_plot_gpu_mask(n, sp_test, bstReg_gpu, scalers, ode_o, ode_n, mask_pred)
            plt.show()
