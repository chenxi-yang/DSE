
from constants import *
from optimization import *
from test import *

# select benchmark program
# from program1 import *
# from program2 import *
# from program3 import *
# from program4 import *
# from program5 import *
# from program6 import *
from program7 import *
# from program8 import *
# from program_test_disjunction import *
# from program_test_disjunction_2 import *

from args import *

def test(X_train, y_train, theta_l, theta_r, target):
    plot_sep_quan_safe_trend(X_train, y_train, theta_l, theta_r, target, k=50)


def evaluation(X_train, y_train, theta_l, theta_r, target, stop_val, epoch=1000, lr=0.00001):
    # # res_theta, loss, loss_list = direct(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000)
    res_theta, loss, loss_list, q, c = gd_direct_noise(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=stop_val, epoch=1000, lr=lr)
    # # res_theta, loss, loss_list = gd_gaussian_noise(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)
    # # res_theta, loss, loss_list = gd(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)

    eval(X_train, y_train, res_theta, target, 'train')
    eval(X_test, y_test, res_theta, target, 'test')



if __name__ == "__main__":
    args = get_args()
    lr = args.lr
    stop_val = args.stop_val

    # data points generation
    target = domain.Interval(safe_l, safe_r)
    X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=50, target_theta=target_theta, test_size=0.33)

    # add for lambda
    # Loss(theta, lambda) = Q(theta) + lambda * C(theta)


    # Eval
    evaluation(X_train, y_train, theta_l, theta_r, target, stop_val=stop_val, lr=lr)
    

    # # TEST
    # test(X_train, y_train, theta_l, theta_r, target)







