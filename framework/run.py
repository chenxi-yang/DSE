
from constants import *
from optimization import *
from test import *

# select benchmark program
# from program1 import *
# from program2 import *
from program3 import *


if __name__ == "__main__":

    target = domain.Interval(safe_l, safe_r)

    X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=50, target_theta=target_theta, test_size=0.33)

    res_theta, loss, loss_list = direct(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000)
    # res_theta, loss, loss_list = gd_direct_noise(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)
    # res_theta, loss, loss_list = gd_gaussian_noise(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)
    # res_theta, loss, loss_list = gd(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)


    eval(X_train, y_train, res_theta, target, 'train')
    eval(X_test, y_test, res_theta, target, 'test')

    # for res_theta in range(690, 710, 1):
    #     res_theta = res_theta / 10.0
    #     print('------Current theta-------', res_theta)
    #     eval(X_train, y_train, res_theta, target, 'train')
    #     eval(X_test, y_test, res_theta, target, 'test')

    # test_theta = 62.67926235491802# random.uniform(theta_l, theta_r)
    # # random.uniform(theta_l, theta_r)
    # print('theta', test_theta)
    # root = construct_syntax_tree(var(test_theta))
    # symbol_table_list = initialization(x_l, x_r)
    # symbol_table_list = root['entry'].execute(symbol_table_list)






