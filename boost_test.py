# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from reactor_ode_delta import ignite_post, data_gen_f
from dataScaling import LogScaler, AtanScaler, NoScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


# create data
def create_data():
    T = np.random.rand(20) * 1000 + 1001
    n_s = np.random.rand(15) * 7.6 + 0.1
    n_l = np.random.rand(30) * 30

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
    # ode_o = ode_o[ode_o[:, -1] == 5e-7]
    # ode_n = ode_n[ode_n[:, -1] == 5e-7]
    ode_o = np.append(ode_o, n_fuel * np.ones((ode_o.shape[0], 1)), axis=1)
    ode_n = np.append(ode_n, n_fuel * np.ones((ode_n.shape[0], 1)), axis=1)
    ode_o = pd.DataFrame(data=ode_o,
                         columns=columns)
    ode_n = pd.DataFrame(data=ode_n,
                         columns=columns)

    # ode_o = ode_o.drop('N2', axis=1)
    ode_o = ode_o.drop('f', axis=1)

    return ode_o, ode_n


def qt_analysis(res):
    qt = res.drop(res.columns.intersection(['N2', 'dt', 'f']), axis=1)
    qt.hist()
    plt.show()
    qt_std = StandardScaler().fit_transform(qt)

    max = qt_std.max(0)
    min = qt_std.min(0)

    schew = sum(qt_std > 1) / qt_std.shape[0] + \
            sum(qt_std < -1) / qt_std.shape[0]

    return schew


def xgb_model_train(df_x, res, mask, species):
    xgb_models = {}
    scalers = {}
    # species = ['O2','H2O','O','HO2']

    for state in {'b:', 'u:'}:
        mask = ~mask

        if mask_train.any():
            df_x_masked = df_x.loc[mask]
            df_y_masked = res.loc[mask]
            # df_y_masked[df_y_masked>0.2]=np.nan
            X_train, X_test, y_train, y_test = train_test_split(df_x_masked, df_y_masked,
                                                                test_size=.1, random_state=42)

            for sp in species:
                # scaler = LogScaler()
                # scaler = AtanScaler()
                # scaler = NoScaler()
                # scaler = MinMaxScaler()
                scaler = MaxAbsScaler()
                # scaler = StandardScaler()

                outlier = 100

                target_train = scaler.fit_transform(y_train[sp].values.reshape(-1, 1))
                # dtrain = xgb.DMatrix(X_train, label=target_train)
                dtrain = xgb.DMatrix(X_train[target_train < outlier], label=target_train[target_train < outlier])

                target_test = scaler.transform(y_test[sp].values.reshape(-1, 1))
                # dtest = xgb.DMatrix(X_test, label=target_test)
                dtest = xgb.DMatrix(X_test[target_test < outlier], label=target_test[target_test < outlier])
                param = {
                    'max_depth': 10,
                    'gamma': 0.3,
                    'eta': 0.25,
                    'silent': 1,
                    'eval_metric': 'mae',
                    'predictor': 'gpu_predictor',
                    'objective': 'gpu:reg:linear'
                }

                num_round = 100
                bst = xgb.train(param, dtrain, num_round,
                                evals=[(dtest, 'test')], early_stopping_rounds=10)

                xgb_models[state + sp] = bst
                scalers[state + sp] = scaler

                print(sp + ':', r2_score(np.exp(bst.predict(dtest)), target_test[target_test < outlier]))

    xgb.plot_importance(bst)

    pickle.dump(xgb_models, open('xgb_' + '_'.join(xgb_models.keys()) + '_models.p', 'wb'))
    return xgb_models


def tot(df, case):
    out = []
    if case == 'C':
        out = df['H2'] + df['H2O']
    if case == 'O':
        out = 2 * df['O2'] + df['OH'] + df['O'] + df['H2O'] + 2 * df['HO2'] + 2 * df['H2O2']
    if case == 'H':
        out = 2 * df['H2'] + df['H'] + df['OH'] + 2 * df['H2O'] + df['HO2'] + 2 * df['H2O2']
    assert (len(out))
    return out


def sp_plot_gpu_mask(n, species, models, scalers, do, do_1, mask):
    plt.figure()
    for state in {'b:', 'u:'}:
        mask = ~mask
        if mask.any():
            test = do_1[mask][species]
            input = do[mask][species]
            dtest = xgb.DMatrix(do)

            # test_target = (test-input)/input
            test_target = test / input

            # sp_pred = np.exp(models[state + species].predict(dtest))
            sp_pred = scalers[state + species].inverse_transform(
                models[state + species].predict(dtest).reshape(-1, 1))
            model_pred = sp_pred[mask]

            # divide res
            sp_pred = model_pred * do[mask][species].values.reshape(-1, 1)
            # sp_pred = sp_pred + do[mask][species].values.reshape(-1, 1)

            # plus res
            # sp_pred = sp_pred[mask] + do[mask][species].values.reshape(-1, 1)

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test.index, test)
            axarr[0].plot(test.index, sp_pred, 'rd', ms=2)
            axarr[0].set_title(species + ':' + str(r2_score(test.values.reshape(-1, 1), sp_pred)))

            # accuracy
            axarr[1].plot((test.values.reshape(-1, 1) - sp_pred.reshape(-1, 1)) / test.values.reshape(-1, 1), 'b')
            axarr[1].plot((test.values.reshape(-1, 1) - input.values.reshape(-1, 1)) / test.values.reshape(-1, 1), 'r')
            axarr[1].set_ylim(-0.1, 0.1)
            axarr[1].set_title(str(n) + '_' + species)

            # learning
            ax2 = axarr[1].twinx()
            ax2.plot(test_target, 'y:')  # target
            ax2.plot(model_pred, 'r:')  # predict
            # ax2.set_ylim(-0.2,0.2)

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
    # df_x = df_x.drop('f', axis=1)
    train_features = columns.drop(['f'])
    df_x = df_x[train_features]
    # df_y = df_y.drop('N2', axis=1)

    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]
    # %%
    pca = PCA(n_components=3)  # pca for 3D visualization

    scaler_pca = NoScaler()
    # train data
    df_x_s = df_x.sample(frac=0.1)  # take 10% data
    train = pca.fit_transform(scaler_pca.transform(df_x_s))
    print('variance:', sum(pca.explained_variance_ratio_))
    train = np.hstack((train, 0 * np.ones((train.shape[0], 1))))  # add a label

    delta = 1
    cube_train = np.round(train / delta)[:, :3]
    ijk = [tuple(i) for i in cube_train.tolist()]
    skt = set(tuple(i) for i in cube_train.tolist())
    len(skt)
    # cube_test = np.round(test/delta)[:,:3]
    # skt_test = set(tuple(i) for i in cube_test.tolist())
    # test_1=np.round(test[test[:,3]==1]/delta)[:,:3]
    # skt_1 = set(tuple(i) for i in test_1.tolist())
    # test_25=np.round(test[test[:,3]==25]/delta)[:,:3]
    # skt_25 = set(tuple(i) for i in test_25.tolist())
    # print('1:',len(skt.intersection(skt_1))/len(skt_1))
    # print('25:',len(skt.intersection(skt_25))/len(skt_25))

    # %%
    # target study
    # res = (df_y - df_x) / df_x
    res = df_y / df_x
    # res = (df_y - df_x)

    # res = df_y
    df_x['C'] = tot(df_x, 'C')
    df_x['tot:O'] = tot(df_x, 'O')
    df_x['tot:H'] = tot(df_x, 'H')

    # df_x['C'] = df_x['H2'] + df_x['H2O']
    #
    # df_x['tot:O'] = 2 * df_x['O2'] + df_x['OH'] + df_x['O'] + df_x['H2O'] \
    #                 + 2 * df_x['HO2'] + 2 * df_x['H2O2']
    #
    # df_x['tot:H'] = 2 * df_x['H2'] + df_x['H'] + df_x['OH'] + 2 * df_x['H2O'] \
    #                 + df_x['HO2'] + 2 * df_x['H2O2']

    mask_train = df_x['HO2'] < 1

    scaler_dict = {'log': LogScaler(),
                   'no': NoScaler(),
                   'mmx': MinMaxScaler(),
                   'mabs': MaxAbsScaler(),
                   'std': StandardScaler()}

    species = ['H2O2']
    # species = ['O2', 'H2O', 'O', 'HO2']
    # species = columns.drop(['dt','N2','f'])

    bstReg_gpu = {}
    scalers = {}

    for state in {'b:', 'u:'}:
        mask_train = ~mask_train

        if mask_train.any():
            df_x_masked = df_x.loc[mask_train]
            df_y_masked = res.loc[mask_train]
            # df_y_masked[df_y_masked>0.2]=np.nan
            X_train, X_test, y_train, y_test = train_test_split(df_x_masked, df_y_masked,
                                                                test_size=.1, random_state=42)

            for sp in species:
                scaler = scaler_dict['no']

                outlier = 100

                target_train = scaler.fit_transform(y_train[sp].values.reshape(-1, 1))
                # dtrain = xgb.DMatrix(X_train, label=target_train)
                dtrain = xgb.DMatrix(X_train[target_train < outlier], label=target_train[target_train < outlier])

                target_test = scaler.transform(y_test[sp].values.reshape(-1, 1))
                # dtest = xgb.DMatrix(X_test, label=target_test)
                dtest = xgb.DMatrix(X_test[target_test < outlier], label=target_test[target_test < outlier])
                param = {
                    'max_depth': 50,
                    'gamma': 0.15,
                    'eta': 0.25,
                    'silent': 1,
                    'eval_metric': 'mae',
                    'predictor': 'gpu_predictor',
                    'objective': 'gpu:reg:linear'
                }

                num_round = 100
                bst = xgb.train(param, dtrain, num_round,
                                evals=[(dtest, 'test')], early_stopping_rounds=10)

                bstReg_gpu[state + sp] = bst
                scalers[state + sp] = scaler

                print(sp + ':', r2_score(np.exp(bst.predict(dtest)), target_test[target_test < outlier]))

    xgb.plot_importance(bst)
    plt.show()

    pickle.dump((bstReg_gpu, scalers), open('xgb_' + '_'.join(bstReg_gpu.keys()) + '_models.p', 'wb'))

    # %%
    for sp_test in species:
        # for n in [.5, 1.4, 2.6, 5, 10, 13, 25]:
        for n in [13, 25]:
            ode_o, ode_n = test_data(1401, n, columns)
            ode_o = ode_o[train_features]
            ode_o['C'] = tot(ode_o, 'C')
            ode_o['tot:O'] = tot(ode_o, 'O')
            ode_o['tot:H'] = tot(ode_o, 'H')
            # mask_pred = (ode_o['H2'] < 0.01) | (ode_o['O2'] < 0.01)
            mask_pred = ode_o['HO2'] < 1
            # mask_pred = ~mask_pred
            sp_plot_gpu_mask(n, sp_test, bstReg_gpu, scalers, ode_o, ode_n, mask_pred)
            plt.show()
