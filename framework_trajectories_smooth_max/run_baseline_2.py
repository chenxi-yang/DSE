#########################
"""
This is for running the second baseline:
1. find the optimized parameter
2. verify the safety
"""
#########################

from constants import *
from optimization_baseline_2 import *
from test_baseline_2 import *

# select benchmark program
from program1 import *
# from program2 import *
# from program3 import *
# from program4 import *
# from program5 import *
# from program6 import *
# from program6_loop import *
# from program7 import *
# from program8 import *
# from program_test_disjunction import *
# from program_test_disjunction_2 import *

from args import *

optimizer = {
    'gd_direct_noise': gd_direct_noise,
    'direct': direct,
    'gd_gaussian_noise': gd_gaussian_noise,
    'gd': gd,
}

def test(X_train, y_train, theta_l, theta_r, target):
    plot_sep_quan_safe_trend(X_train, y_train, theta_l, theta_r, target, k=50)

if __name__ == "__main__":
    args = get_args()
    lr = args.lr
    stop_val = args.stop_val
    optimizer_name = args.optimizer
    optimize_f = optimizer[optimizer_name]

    # data points generation
    # for i in range(10):
    #     target = domain.Interval(safe_l, safe_r)
    #     X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=10000, target_theta=target_theta, test_size=0.99)

    #     theta, loss, loss_list, q, c = optimize_f(X_train, y_train, theta_l, theta_r, target, lambda_=var(50.0), stop_val=stop_val, epoch=500, lr=lr)
    #     # theta = var(5.4852)

    #     eval(X_train, y_train, theta, target, 'train')
    #     eval(X_test, y_test, theta, target, 'test')

    # test
    # target = domain.Interval(safe_l, safe_r)
    # X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=100, target_theta=target_theta, test_size=0.33)

    # test(X_train, y_train, theta_l, theta_r, target)




