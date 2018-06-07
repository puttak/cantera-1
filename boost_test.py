import xgboost as xgb
# read in data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

from reactor_ode_delta import ignite_post
import pandas as pd


# test=ode_o.drop('N2',axis=1)
def sp_plot(species, models, do, do_1):
    test = do_1[species]
    sp_pred = models[species].predict(do)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(test)
    plt.plot(sp_pred, 'rd')
    plt.title(species + ':' + str(r2_score(test, sp_pred)))

    plt.subplot(2, 2, 1)
    plt.plot((test - sp_pred) / test)


def sp_plot_gpu(species, models, do, do_1):
    test = do_1[species]
    dtest = xgb.DMatrix(do)
    # sp_pred = models[species].predict(dtest)
    sp_pred=np.exp(models[species].predict(dtest))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(test)
    plt.plot(sp_pred, 'rd',ms=2)
    plt.title(species + ':' + str(r2_score(test, sp_pred)))

    plt.subplot(1, 2, 2)
    plt.plot((test - sp_pred) / test)
    plt.ylim(-0.1, 0.1)
    plt.show()
    return sp_pred


def sp_plot_gpu_mask(species, models, do, do_1,mask_pred):
    test = do_1[mask_pred][species]
    dtest = xgb.DMatrix(do)
    # sp_pred = models[species].predict(dtest)
    sp_pred = np.exp(models[species].predict(dtest))
    # sp_pred=np.ma.masked_where(mask_pred,sp_pred)
    sp_pred = sp_pred[mask_pred]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(test.index,test)
    plt.plot(test.index,sp_pred, 'rd',ms=2)
    plt.title(species + ':' + str(r2_score(test, sp_pred)))

    plt.subplot(1, 2, 2)
    plt.plot((test - sp_pred) / test)
    plt.ylim(-0.1, 0.1)
    plt.show()



# load training
df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))

# load test
temp = 1501
n_fuel = 6
ode_o, ode_n = ignite_post((temp, n_fuel, 'H2'))
ode_o = np.asarray(ode_o)
ode_n = np.asarray(ode_n)
ode_o = ode_o[ode_o[:, -1] == 1e-6]
ode_n = ode_n[ode_n[:, -1] == 1e-6]

ode_o = pd.DataFrame(data=ode_o,
                     columns=df_x.columns)
ode_n = pd.DataFrame(data=ode_n,
                     columns=df_x.columns)

ode_o = ode_o.drop('N2', axis=1)
ode_n = ode_n.drop('N2', axis=1)
df_x = df_x.drop('N2', axis=1)
df_y = df_y.drop('N2', axis=1)
# df_x.head()
# df_x['C']=df_x['H2']+df_x['H2O']
# ode_o['C']=ode_o['H2']+ode_o['H2O']
# get sum
tot = df_x[df_x.columns.drop(['T', 'dt'])]
tot.head()
tot.sum(axis=1)

bstReg_gpu = {}
# for sp in df_y.columns.drop('dt'):
# mask=df_x['H2O2']>1e-7
# mask=df_x['H2O2']<1e-7
# mask = df_x['O2'] < 0.01
mask = (df_x['O2'] < 0.01) | (df_x['H2'] < 0.01)
mask = ~mask
for sp in ['T']:
    #     dtrain=xgb.DMatrix(df_x.loc[mask],label=df_y.loc[mask][sp]*10e10)
    # target = df_y.loc[mask][sp]
    target=np.log(df_y.loc[mask][sp])
    dtrain = xgb.DMatrix(df_x.loc[mask], label=target)
    param = {
        # 'ntree_limit':200,
        'max_depth': 10,
        'eta': 0.3,
        'silent': 1,
        'predictor': 'gpu_predictor',
        'objective': 'gpu:reg:linear'

    }

    num_round = 100
    bst = xgb.train(param, dtrain, num_round)

    bstReg_gpu[sp] = bst

    print(sp + ':', r2_score(np.exp(bst.predict(dtrain)), target))

pred = sp_plot_gpu('T', bstReg_gpu, ode_o, ode_n)
# xgb.plot_importance(bst)
# plt.plot(np.ma.masked_where((ode_o['H2']<0.01)|(ode_o['O2']<0.01),pred),'rd')
# plt.show()
mask_pred = (ode_o['T']<0.01)|(ode_o['O2']<0.01)
mask_pred = ~mask_pred
sp_plot_gpu_mask('T', bstReg_gpu, ode_o, ode_n,mask_pred)
plt.show()