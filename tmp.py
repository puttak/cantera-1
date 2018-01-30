import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import cantera as ct
#
# print("Running Cantera version: {}".format(ct.__version__))


def data_scaling(input, case, norm=None, std=None):

    switcher={
        'std': 'std',
        'nrm': 'nrm',
        'log':'log'
    }

    if switcher.get(case) == 'std':
        if not norm:
            # print(1)
            norm = MinMaxScaler()
            std = StandardScaler()
            out = std.fit_transform(input)
            out = norm.fit_transform(out)
        else:
            # print(2)
            out = std.transform(input)
            out = norm.transform(out)

    if switcher.get(case) == 'nrm':
        if not norm:
            # print(1)
            norm = MinMaxScaler()
            std = StandardScaler()
            out = norm.fit_transform(input)
        else:
            # print(2)
            out = norm.transform(input)

    if switcher.get(case) == 'log':
        out = np.log(np.asarray(input)+1e-20)
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



    return out, norm, std


def data_inverse(input,case, norm, std):

    switcher={
        'std':'std',
        'nrm':'nrm',
        'log':'log'
    }

    if switcher.get(case) == 'std':
        out = norm.inverse_transform(input)
        out = std.inverse_transform(out)

    if switcher.get(case) == 'nrm':
        out = norm.inverse_transform(input)

    if switcher.get(case) == 'log':
        out = norm.inverse_transform(input)
        out = np.exp(out)
    # if switcher.get(case) == 'log':
    #     out = std.inverse_transform(input)
    #     out = np.exp(out)
    #     out = norm.inverse_transform(out)

    return np.double(out)

