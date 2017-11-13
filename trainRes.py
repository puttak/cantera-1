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
from keras.layers import Dense, Input
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from res_block import res_block



import cntk

from reactor_ode import train_org, train_new, train_res
from sklearn.utils import shuffle

# prepare data
# train_org, train_new = shuffle(train_org, train_new)

train_org = train_org[0.000:0.001]
train_new = train_new[0.000:0.001]

output = train_org.columns
label_values = []
input_norm_scalers = {}
input_std_scalers = {}
for itm in output:
    print(itm)
    norm_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    # same input_label scaler
    # scaler.fit(np.concatenate([train_org[itm], train_new[itm]], axis=0).reshape(-1, 1))
    # out = scaler.transform(train_org[itm].values.reshape(-1, 1))
    # input scaler
    out = std_scaler.fit_transform(train_org[itm].values.reshape(-1, 1))
    out = out/out.max()
    #out = norm_scaler.fit_transform(out)
    label_values.append(out)
    input_norm_scalers[itm] = norm_scaler
    input_std_scalers[itm] = std_scaler

x_train = np.concatenate(
    label_values,
    axis=1)

output = train_new.columns
label_values = []
label_norm_scalers = {}
label_std_scalers = {}
for itm in output:
    print(itm)
    # same input_label scaler
    # out = input_scalers[itm].transform(train_new[itm].values.reshape(-1, 1))
    # input scaler
    norm_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    out = std_scaler.fit_transform(train_new[itm].values.reshape(-1, 1))
    out = out/out.max()
    #out = norm_scaler.fit_transform(out)

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
batch_size = 128
epochs = 500
vsplit = 0.1
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,), dtype='float32')
print(inputs.dtype)
# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

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
a = (predict - y_train) / (y_train)
error = abs(pd.DataFrame(data=a, columns=train_org.columns))

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
