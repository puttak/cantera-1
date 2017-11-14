import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_floatx('float32')
print(K.floatx())

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from res_block import res_block

import cntk

from reactor_ode import train_org, train_new, train_res
from sklearn.utils import shuffle

# prepare data
# train_org = train_org[0.000:0.001]
# train_new = train_new[0.000:0.001]
train_org, train_new = shuffle(train_org, train_new)

output = train_org.columns
label_values = []
input_norm_scalers = {}
input_std_scalers = {}
for itm in output:
    print(itm)
    norm_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    # same input_label scaler
    # std_scaler.fit(np.concatenate([train_org[itm], train_new[itm]], axis=0).reshape(-1, 1))
    # out = std_scaler.transform(train_org[itm].values.reshape(-1, 1))
    # input scaler
    out = std_scaler.fit_transform(train_org[itm].values.reshape(-1, 1))
    # out = out/out.max()
    out = 2 * norm_scaler.fit_transform(out) - 1

    label_values.append(out)
    input_norm_scalers[itm] = norm_scaler
    input_std_scalers[itm] = std_scaler

x_train = np.concatenate(
    label_values,
    axis=1)

output = train_new.columns
# output = ['H2']
label_values = []
label_norm_scalers = {}
label_std_scalers = {}
for itm in output:
    print(itm)
    # same input_label scaler
    # out = input_std_scalers[itm].transform(train_new[itm].values.reshape(-1, 1))
    # input scaler
    norm_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    # out = train_new[itm].values.reshape(-1, 1)
    out = std_scaler.fit_transform(train_new[itm].values.reshape(-1, 1))
    # out = out-out.min()
    out = 2 * norm_scaler.fit_transform(out) - 1

    label_values.append(out)
    label_norm_scalers[itm] = norm_scaler
    label_std_scalers[itm] = std_scaler

y_train = np.concatenate(
    label_values,
    axis=1
)

######################
print('set up ANN')
# ANN parameters
dim_input = x_train.shape[1]
dim_label = y_train.shape[1]
n_neuron = 100
batch_size = 512
epochs = 500
vsplit = 0.1
batch_norm = True

# This returns a tensor
inputs = Input(shape=(dim_input,), dtype='float32')
print(inputs.dtype)
# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, name='1_base')(inputs)
# x = BatchNormalization(axis=-1, name='1_base_bn')(x)
x = Activation('relu')(x)

# less then 2 res_block, there will be variance
x = res_block(x, n_neuron, stage=1, block='a', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='b', bn=batch_norm)

# x = res_block(x, n_neuron, stage=1, block='c', bn=batch_norm)
# x = res_block(x, n_neuron, stage=1, block='d', bn=batch_norm)

predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.cntk.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)
callbacks_list = [checkpoint]

# fit the model
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True)

# loss
fig = plt.figure()
plt.semilogy(history.history['loss'])
if vsplit:
    plt.semilogy(history.history['val_loss'])
plt.title('mae')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')


#########################################
model.load_weights("./tmp/weights.best.cntk.hdf5")

predict = model.predict(x_train)
df_y_prdt = pd.DataFrame(data=predict, columns=output)

error = (predict - y_train) / y_train
df_error = abs(pd.DataFrame(data=error, columns=output))

y_prdt_inv = []
for itm in output:
    print(itm)
    out = label_norm_scalers[itm].inverse_transform(0.5 * (df_y_prdt[itm].values.reshape(-1, 1) + 1))
    out = label_std_scalers[itm].inverse_transform(out)
    y_prdt_inv.append(out)
y_prdt_inv = np.concatenate(
    y_prdt_inv,
    axis=1
)
df_y_prdt_inv = pd.DataFrame(data=y_prdt_inv, columns=output)

error_inv = (y_prdt_inv - train_new) / (train_new+1e-10)
df_error_inv = abs(pd.DataFrame(data=error_inv, columns=output))


def acc_plt(sp):
    plt.figure()
    plt.plot(train_new[sp],df_y_prdt_inv[sp],'kd',ms=1)
    plt.axis('tight')
    plt.axes().set_aspect('equal')
    #plt.axis([train_new[sp].min(), train_new[sp].max(), train_new[sp].min(), train_new[sp].max()], 'tight')
    plt.title(sp)

acc_plt('H2O2')

# predict = model.predict(x_train[0])
# predict[0,18]
# cntk.combine(model.outputs).save('mayerTest.dnn')
#
# p_loc = [34, 38, 42, 46, 50]
# for p_iso in p_loc:
#     ref = df.loc[df['p'] == p_iso]
#     x_test = np.append(
#         p_scaler.transform(ref['p'].values.reshape(-1, 1)),
#         he_scaler.transform(ref['he'].values.reshape(-1, 1)),
#         axis=1)
#
#     predict = model.predict(x_test.astype(float))
#     for i, itm in enumerate(output):
#         plt.figure()
#         plt.plot(ref['T'], ref[itm], 'r:')
#         plt.plot(ref['T'], label_scalers[itm].inverse_transform(predict[:, i].reshape(-1, 1)), 'b-')
#         plt.title(itm+'@'+str(p_iso))
#         print(
#             itm, (
#                 label_scalers[itm].data_max_,
#                 label_scalers[itm].data_min_,
#                 label_scalers[itm].data_range_)
#         )
#
# print(
#     p_scaler.data_min_,
#     p_scaler.data_max_,
#     p_scaler.data_range_,
#     he_scaler.data_min_,
#     he_scaler.data_max_,
#     he_scaler.data_range_
#
