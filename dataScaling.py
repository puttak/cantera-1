import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, QuantileTransformer


class dataScaling(object):
    def __init__(self):
        self.norm = None
        self.norm_1 = None
        self.std = None
        self.case = None
        self.scale = 1

        self.switcher = {
            'std': 'std',
            'std2': 'std2',
            'nrm': 'nrm',
            'no':'no',
            'log': 'log',
            'log_std':'log_std',
            'log2': 'log2',
            'tan': 'tan'
        }

    def fit_transform(self, input_data, case):
        self.case = case
        if self.switcher.get(self.case) == 'std':
            # self.norm = MaxAbsScaler()
            self.norm = MinMaxScaler()
            self.norm_1 = MinMaxScaler()
            self.std = StandardScaler()
            out = self.norm_1.fit_transform(input_data)
            out = self.std.fit_transform(out)
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == 'std2':
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)

        if self.switcher.get(self.case) == 'std_nrm':
            # self.norm = MaxAbsScaler()
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == 'nrm':
            self.norm = MinMaxScaler()
            # self.norm = MaxAbsScaler()
            self.std = StandardScaler()
            out = self.norm.fit_transform(input_data)

        if self.switcher.get(self.case) == 'no':
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = input_data

        if self.switcher.get(self.case) == 'log':
            out = - np.log(np.asarray(input_data / self.scale) + 1e-20)
            # out = - np.log(np.asarray(input_data) + 1e-20)
            # self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'log_std':
            out = - np.log(np.asarray(input_data / 100) + 1e-20)
            # out = - np.log(np.asarray(input_data + 1e-20))
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(out)
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == 'log2':
            self.norm = MinMaxScaler()
            self.norm_1 = MinMaxScaler()
            out = self.norm.fit_transform(input_data)
            out = np.log(np.asarray(out) + 1e-20)
            out = self.norm_1.fit_transform(out)

        if self.switcher.get(self.case) == 'tan':
            # self.norm = MinMaxScaler()
            self.norm = MaxAbsScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)
            out = self.norm.fit_transform(out)
            # out = np.tan((2 * np.asarray(out) - 1) / (2 * np.pi + 1e-20))
            out = np.tan(out / (2 * np.pi + 1e-20))

        return out

    def transform(self, input_data):
        if self.switcher.get(self.case) == 'std':
            out = self.norm_1.transform(input_data)
            out = self.std.transform(out)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == 'std2':
            out = self.std.transform(input_data)

        if self.switcher.get(self.case) == 'std_nrm':
            out = self.std.transform(input_data)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == 'nrm':
            out = self.norm.transform(input_data)

        if self.switcher.get(self.case) == 'no':
            out = input_data

        if self.switcher.get(self.case) == 'log':
            out = - np.log(np.asarray(input_data / self.scale) + 1e-20)
            # out = - np.log(np.asarray(input_data) + 1e-20)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'log_std':
            out = - np.log(np.asarray(input_data / 100) + 1e-20)
            # out = - np.log(np.asarray(input_data + 1e-20))
            out = self.std.transform(out)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == 'log2':
            out = self.norm.transform(input_data)
            out = np.log(np.asarray(out) + 1e-20)
            out = self.norm_1.transform(out)

        if self.switcher.get(self.case) == 'tan':
            out = self.std.transform(input_data)
            out = self.norm.transform(out)
            # out = np.tan((2 * np.asarray(out) - 1) / (2 * np.pi + 1e-20))
            out = np.tan(out / (2 * np.pi + 1e-20))

        return out

    def inverse_transform(self, input_data):

        if self.switcher.get(self.case) == 'std':
            out = self.norm.inverse_transform(input_data)
            out = self.std.inverse_transform(out)
            out = self.norm_1.inverse_transform(out)

        if self.switcher.get(self.case) == 'std2':
            out = self.std.inverse_transform(input_data)

        if self.switcher.get(self.case) == 'std_nrm':
            out = self.norm.inverse_transform(input_data)
            out = self.std.inverse_transform(out)

        if self.switcher.get(self.case) == 'nrm':
            out = self.norm.inverse_transform(input_data)

        if self.switcher.get(self.case) == 'no':
            out = input_data

        if self.switcher.get(self.case) == 'log':
            out = self.std.inverse_transform(input_data)
            out = (np.exp(-out) - 1e-20) * self.scale

        if self.switcher.get(self.case) == 'log_std':
            out = self.norm.inverse_transform(input_data)
            out = self.std.inverse_transform(out)
            # out = np.exp(-out) -1e-20
            out = (np.exp(-out) - 1e-20) * 100

        if self.switcher.get(self.case) == 'log2':
            out = self.norm_1.inverse_transform(input_data)
            out = np.exp(out) - 1e-20
            out = self.norm.inverse_transform(out)

        if self.switcher.get(self.case) == 'tan':
            # out = (2 * np.pi * np.arctan(input_data) + 1) / 2
            out = (2 * np.pi + 1e-20) * np.arctan(input_data)
            out = self.norm.inverse_transform(out)
            out = self.std.inverse_transform(out)

        return out


class LogScaler(object):

    def fit_transform(self, input_data):
        out = np.log(input_data)
        return out

    def transform(self, input_data):
        out = np.log(input_data)
        return out

    def inverse_transform(self, input_data):
        out = np.exp(input_data)
        return out


class LogMirrorScaler(object):

    def fit_transform(self, input_data):
        out = np.log(input_data)
        return out

    def transform(self, input_data):
        out = np.log(input_data)
        return out

    def inverse_transform(self, input_data):
        out = np.exp(input_data)
        return out


class AtanScaler(object):

    def fit_transform(self, input_data):
        out = np.arctan(input_data)
        return out

    def transform(self, input_data):
        out = np.arctan(input_data)
        return out

    def inverse_transform(self, input_data):
        out = np.tan(input_data)
        return out


class NoScaler(object):

    def fit(self, input_data):
        out = input_data
        return out

    def fit_transform(self, input_data):
        out = input_data
        return out

    def transform(self, input_data):
        out = input_data
        return out

    def inverse_transform(self, input_data):
        out = input_data
        return out


