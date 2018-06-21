import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

from reactor_ode_delta import ignite_post, data_gen_f
import pandas as pd
from deltaNets import combustionML
from boost_test import test_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

if __name__ == '__main__':
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
    # res = (df_y - df_x) / df_x
    # res = (df_y - df_x)
    res = df_y
    # res = res.drop(res.columns.intersection(['f', 'N2', 'dt']), axis=1)
    # species = ['H2O']
    species = ['O2','O','H2O','H2O2']
    # target = res[species]
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = MaxAbsScaler()
    target = pd.DataFrame(scaler.fit_transform(res[species]),columns=species)

    df_x['C'] = df_x['H2'] + df_x['H2O']

    # df_x['tot:O'] = 2 * df_x['O2'] + df_x['OH'] + df_x['O'] + df_x['H2O'] \
    #                 + 2 * df_x['HO2'] + 2 * df_x['H2O2']
    #
    # df_x['tot:H'] = 2 * df_x['H2'] + df_x['H'] + df_x['OH'] + 2 * df_x['H2O'] \
    #                 + df_x['HO2'] + 2 * df_x['H2O2']

    #%%
    nn_std = combustionML(df_x, target, 'nrm')
    r2 = nn_std.run([200, 2, 0.5])

    #%%
    # test
    # species=res.columns

    for sp in species:
        for n in [1,25]:
            ode_o, ode_n = test_data(1501, n, columns)
            ode_o['C'] = ode_o['H2'] + ode_o['H2O']
            # ode_o['tot:O'] = 2 * ode_o['O2'] + ode_o['OH'] + ode_o['O'] + ode_o['H2O'] \
            #                  + 2 * ode_o['HO2'] + 2 * ode_o['H2O2']
            # ode_o['tot:H'] = 2 * ode_o['H2'] + ode_o['H'] + ode_o['OH'] + 2 * ode_o['H2O'] \
            #                  + ode_o['HO2'] + 2 * ode_o['H2O2']

            # pred=pd.DataFrame(nn_std.inference(ode_o),columns=res.columns)
            # pred=pd.DataFrame(nn_std.inference(ode_o),columns=target.columns)
            pred=pd.DataFrame(scaler.inverse_transform(nn_std.inference(ode_o)),columns=target.columns)

            test=(ode_o[sp]+pred[sp]*ode_o[sp]-ode_n[sp])/ode_n[sp]


            f, axarr = plt.subplots(1,2)
            # plt.subplot(1, 2, 1)
            axarr[0].plot(ode_n[sp])
            # axarr[0].plot(ode_o[sp] + pred[sp] * ode_o[sp], 'rd', ms=2)
            axarr[0].plot(pred[sp], 'rd', ms=2)
            # axarr[0].set_title(sp + ':' + str(r2_score(test.values.reshape(-1,1), sp_pred)))

            # plt.subplot(1, 2, 2)
            # _,ax1=plt.subplots()
            # axarr[1].plot((ode_n[sp] - ode_o[sp] - pred[sp] * ode_o[sp]) / ode_n[sp])
            axarr[1].plot((ode_n[sp] - pred[sp]) / ode_n[sp])
            axarr[1].set_ylim(-0.1, 0.1)
            axarr[1].set_title(str(n) + '_' + sp)


            ax2 = axarr[1].twinx()
            ax2.plot((ode_n[sp] - ode_o[sp]) / ode_o[sp],'y:')
            ax2.plot(pred[sp],'r:')
            ax2.set_ylim(-0.2,0.2)

            plt.show()