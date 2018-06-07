import numpy as np

np.random.seed(8)

def redball_blueball(count=1):
    '''
    generate a red or a blue ball
    :param count:
    :return:
    '''

    bag = np.random.random(count)

    return ['red' if (x <= 0.4) else 'blue' for x in bag.tolist()]



np.random.seed(8)

bag_10 = redball_blueball(10)
bag_100 = redball_blueball(100)
bag_1000 = redball_blueball(1000)
bag_10000 = redball_blueball(10000)

from collections import Counter

def pball(ball_list):
    c_dict=Counter(ball_list)
    tmp=np.asarray([c_dict['red'],c_dict['blue']])
    return tmp/tmp.sum()



from scipy import stats

KL=stats.entropy(pk=pball(bag_10),qk=[0.4,0.6])

import matplotlib.pyplot as plt
x_axis = np.arange(-10,10,0.001)
dist_a = stats.norm.pdf(x_axis,0,2)
dist_b = stats.norm.pdf(x_axis,1,2)
plt.plot(x_axis,dist_a)
plt.plot(x_axis,dist_b)
plt.fill_between(x_axis,dist_a, dist_b, where=dist_b>dist_a, facecolor='green', interpolate=True)
plt.fill_between(x_axis,dist_a, dist_b, where=dist_b<dist_a, facecolor='blue', interpolate=True)
plt.show()

actual = np.array([0.4, 0.6])
model_1 = pball(bag_10)
model_2 = pball(bag_100)

kl_1 = (model_1*np.log(model_1/actual)).sum()
kl_2 = (model_2*np.log(model_2/actual)).sum()


