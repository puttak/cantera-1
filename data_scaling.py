import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, QuantileTransformer


def data_scaling(input_data, case, norm=None, std=None):
    switcher = {
        'std': 'std',
        'nrm': 'nrm',
        'log': 'log',
        'tan': 'tan'
    }

    if switcher.get(case) == 'std':
        if not norm:
            norm = MaxAbsScaler()
            std = StandardScaler()
            out = std.fit_transform(input_data)
            out = norm.fit_transform(out)
        else:
            out = std.transform(input_data)
            out = norm.transform(out)

    if switcher.get(case) == 'nrm':
        if not norm:
            norm = MinMaxScaler()
            std = StandardScaler()
            out = norm.fit_transform(input_data)
        else:
            out = norm.transform(input_data)

    if switcher.get(case) == 'log':
        out = np.log(np.asarray(input_data) + 1e-20)
        if not norm:
            norm = MinMaxScaler()
            std = StandardScaler()
            out = norm.fit_transform(out)
        else:
            out = norm.transform(out)
    # if switcher.get(case) == 'log':
    #
    #     if not norm:
    #         norm = MinMaxScaler()
    #         std = MinMaxScaler()
    #         out = norm.fit_transform(input)
    #         out = np.log(np.asarray(out) + 1e-20)
    #         out = std.fit_transform(out)
    #     else:
    #         out = norm.transform(input)
    #         out = np.log(np.asarray(out) + 1e-20)
    #         out = std.transform(out)
    if switcher.get(case) == 'tan':

        if not norm:
            norm = MinMaxScaler()
            std = StandardScaler()
            out = norm.fit_transform(input_data)

        else:
            out = norm.transform(input_data)
        out = np.tan((2 * np.asarray(out) - 1) / (2 * np.pi + 1e-20))

    return out, norm, std


def data_inverse(input_data, case, norm, std):
    switcher = {
        'std': 'std',
        'nrm': 'nrm',
        'log': 'log',
        'tan': 'tan'
    }

    if switcher.get(case) == 'std':
        out = norm.inverse_transform(input_data)
        out = std.inverse_transform(out)

    if switcher.get(case) == 'nrm':
        out = norm.inverse_transform(input_data)

    if switcher.get(case) == 'log':
        out = norm.inverse_transform(input_data)
        out = np.exp(out)
    # if switcher.get(case) == 'log':
    #     out = std.inverse_transform(input)
    #     out = np.exp(out)
    #     out = norm.inverse_transform(out)
    if switcher.get(case) == 'tan':
        # out = norm.inverse_transform(input)
        out = (2 * np.pi * np.arctan(input_data) + 1) / 2
        out = norm.inverse_transform(out)

    return out
