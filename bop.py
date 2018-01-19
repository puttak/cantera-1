import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_lowercase

from sklearn import model_selection, metrics

# import cntk
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_floatx('float32')
print("precision: " + K.floatx())

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from res_block import res_block
from reactor_ode_p import data_gen, data_scaling, data_inverse

# T = np.linspace(1501, 2001, 15)
# n = np.linspace(1, 0., 15)
# XX, YY = np.meshgrid(T, n)
# ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)
#
# df_x_input, df_y_target = data_gen(ini, 'CH4')

# x_input, y_target = shuffle(x_input, y_target)
# y_target = y_target.drop('temperature', axis=1)

#x_input, norm_x, std_x = data_scaling(df_x_input)
#y_target, norm_y, std_y = data_scaling(df_y_target)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_target,
#                                                                     test_size=0.1,
#                                                                     random_state=42)


# x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x_input, df_y_target,
#                                                                     test_size=0.1,
#                                                                     random_state=42)
# x_train, norm_x, std_x = data_scaling(x_train)
# y_train, norm_y, std_y = data_scaling(y_train)
# x_test, _, _ = data_scaling(x_test,norm_x,std_x)
# y_test, _, _ = data_scaling(y_test,norm_y,std_y)

class combustionML(object):
    # def __init__(self, df_x_input, df_y_target, x_train, x_test, y_train, y_test):
    def __init__(self, df_x_input, df_y_target):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x_input, df_y_target,
                                                                    test_size=0.3,
                                                                    random_state=42)
        self.x_train, self.norm_x, self.std_x = data_scaling(x_train)
        self.y_train, self.norm_y, self.std_y = data_scaling(y_train)
        x_test, _, _ = data_scaling(x_test,self.norm_x,self.std_x)
        #y_test, _, _ = data_scaling(y_test,norm_y,std_y)

        self.df_x_input = df_x_input
        self.df_y_target = df_y_target
        self.x_test = pd.DataFrame(data=x_test, columns=df_x_input.columns)
        self.y_test = pd.DataFrame(data=y_test, columns=df_y_target.columns)

        self.model = None
        self.history = None
        self.callbacks_list = None
        self.vsplit = None
        self.predict = None


    def composeResnetModel(self, n_neurons=200, blocks=2, drop1=0.1, loss='mse', optimizer='adam', batch_norm=False):
        ######################
        print('set up ANN')
        # ANN parameters
        dim_input = self.x_train.shape[1]
        dim_label = self.y_train.shape[1]

        # This returns a tensor
        inputs = Input(shape=(dim_input,), dtype='float32')
        print(inputs.dtype)
        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(n_neurons, name='1_base')(inputs)
        # x = BatchNormalization(axis=-1, name='1_base_bn')(x)
        x = Activation('relu')(x)

        # less then 2 res_block, there will be variance
        for b in range(blocks):
            x = res_block(x, n_neurons, stage=1, block=ascii_lowercase[b], d1=drop1, bn=batch_norm)

        predictions = Dense(dim_label, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=predictions)

        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # checkpoint (save the best model based validate loss)
        filepath = "./tmp/weights.best.cntk.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=10)
        self.callbacks_list = [checkpoint]

    def fitModel(self, batch_size=1024, epochs=400, vsplit=0.2):
        # fit the model
        self.vsplit = vsplit
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=vsplit,
            verbose=2,
            callbacks=self.callbacks_list,
            shuffle=True)

    def prediction(self):
        self.model.load_weights("./tmp/weights.best.cntk.hdf5")

        predict = self.model.predict(self.x_test.values)
        predict = data_inverse(predict, self.norm_y, self.std_y)
        self.predict = pd.DataFrame(data=predict, columns=self.df_y_target.columns)

        R2_score = -abs(metrics.r2_score(predict, self.y_test))
        print(R2_score)
        return R2_score

    def inference(self,x):
        #self.model.load_weights("./tmp/weights.best.cntk.hdf5")

        tmp,_,_=data_scaling(x, self.norm_x,self.std_x)
        predict = self.model.predict(tmp)
        out = data_inverse(predict, self.norm_y, self.std_y)

        return out

    def acc_plt(self, sp):
        plt.figure()
        plt.plot(self.y_test[sp], self.predict[sp], 'kd', ms=1)
        plt.axis('tight')
        plt.axis('equal')
        # plt.axis([train_new[sp].min(), train_new[sp].max(), train_new[sp].min(), train_new[sp].max()], 'tight')
        r2 = round(metrics.r2_score(self.y_test[sp],self.predict[sp]),6)
        plt.title(sp+' : r2 = '+str(r2))

    # loss
    def plt_loss(self):
        # fig = plt.figure()
        plt.semilogy(self.history.history['loss'])
        if self.vsplit:
            plt.semilogy(self.history.history['val_loss'])
        plt.title('mae')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

    def run(self, hyper):
        print(hyper)

        self.composeResnetModel(n_neurons=hyper[0], blocks=hyper[1], drop1=hyper[2])
        self.fitModel(epochs=1000, batch_size=1024 * 8)

        return self.prediction()


if __name__ == "__main__":
    T = np.linspace(1001, 3101, 100)
    n = np.linspace(4, 0., 100)
    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

    df_x_input, df_y_target = data_gen(ini, 'H2')

    test = combustionML(df_x_input, df_y_target)

    bop = False
    if bop:
        from skopt import gp_minimize

        res = gp_minimize(test.run,  # the function to minimize
                          [(100, 200, 300, 400, 500, 600),
                           (2, 5),
                           (0., 0.1, 0.2, 0.3, 0.4, 0.5)],  # the bounds on each dimension of x

                          acq_func="EI",  # the acquisition function
                          n_calls=15,  # the number of evaluations of f
                          n_random_starts=5,  # the number of random initialization points
                          random_state=123)  # the random seed

        from skopt.plots import plot_convergence

        plot_convergence(res);
        print(res.x, res.fun)

    test.run([400, 4, 0.1])
    # test.composeResnetModel(400,2,0.)
    # test.prediction()
    sp = 'O2'
    plt.plot(test.x_test[sp], test.y_test[sp], 'kd', ms=1)
    test.acc_plt(sp)
