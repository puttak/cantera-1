import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint

import cntk

from reactor_ode import train_input, timeHistory
from sklearn.utils import shuffle


def res_block(input_tensor, n_neuron, stage, block, bn=False):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Dense(n_neuron, name=conv_name_base + '2a')(input_tensor)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=conv_name_base + '2b')(x)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x

# prepare data
train_input = train_input.loc[:, (train_input != 0).any(axis=0)]
timeHistory = timeHistory.loc[:, (timeHistory != 0).any(axis=0)]

train_input,timeHistory = shuffle(train_input,timeHistory)


output = train_input.columns
label_values = []
label_scalers = {}
for itm in output:
    print(itm)
    scaler = preprocessing.MinMaxScaler()
    out = scaler.fit_transform(train_input[itm].values.reshape(-1, 1))
    label_scalers[itm] = scaler
    label_values.append(out)

x_train = np.concatenate(
    label_values,
    axis=1)

output = timeHistory.columns
label_values = []

for itm in output:
    print(itm)
    out = label_scalers[itm].transform(timeHistory[itm].values.reshape(-1, 1))
    label_values.append(out)

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
batch_size = 1024
epochs = 4000
vsplit = 0.1
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, n_neuron, stage=1, block='a', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='b', bn=batch_norm)

# x = res_block(x, n_neuron, stage=1, block='c', bn=batch_norm)
# x = res_block(x, n_neuron, stage=1, block='d', bn=batch_norm)

predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

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
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

#########################################
model.load_weights("./tmp/weights.best.cntk.hdf5")

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
# )
