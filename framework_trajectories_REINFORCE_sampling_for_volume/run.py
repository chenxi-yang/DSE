
from constants import *
from optimization import *
from test import *

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


def evaluation(X_train, y_train, theta_l, theta_r, target, lambda_, stop_val, epoch=500, lr=0.00001):
    # # res_theta, loss, loss_list = direct(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000)
    res_theta, loss, loss_list, q, c = optimize_f(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=stop_val, epoch=500, lr=lr)
    # # res_theta, loss, loss_list = gd_gaussian_noise(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)
    # # res_theta, loss, loss_list = gd(X_train, y_train, theta_l, theta_r, target, stop_val=1.0, epoch=1000, lr=0.1)

    eval(X_train, y_train, res_theta, target, 'train')
    eval(X_test, y_test, res_theta, target, 'test')


def best_lambda(X_train, y_train, theta):
    c = cal_c(X_train, y_train, theta)
    q = cal_q(X_train, y_train, theta)

    if c.data.item() <= 0.0:
        res_lambda = var(0.0)
    else:
        res_lambda = B
    return res_lambda, q.add(res_lambda.mul(c)) # lambda, L_max


def best_theta(X_train, y_train, lambda_):
    theta, loss, loss_list, q, c = optimize_f(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=stop_val, epoch=500, lr=lr)

    return theta, loss


if __name__ == "__main__":
    args = get_args()
    lr = args.lr
    stop_val = args.stop_val
    t_epoch = args.t_epoch
    optimizer_name = args.optimizer
    optimize_f = optimizer[optimizer_name]
    w = args.w

    # data points generation
    target = domain.Interval(safe_l, safe_r)
    X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=10000, target_theta=target_theta, test_size=0.99)

    # add for lambdas
    # Loss(theta, lambda) = Q(theta) + lambda * C(theta)

    for i in range(5):
        lambda_list = list()
        theta_list = list()
        q = var(0.0)

        for t in range(t_epoch):
            new_lambda = B.mul(q.exp().div(var(1.0).add(q.exp())))

            # BEST_theta(lambda)
            theta, loss, loss_list, q, c = optimize_f(X_train, y_train, theta_l, theta_r, target, lambda_=new_lambda, stop_val=stop_val, epoch=500, lr=lr)
            
            lambda_list.append(new_lambda)
            theta_list.append(theta)

            theta_t = var(0.0)
            for i in theta_list:
                theta_t = theta_t.add(i)
            theta_t = theta_t.div(var(len(theta_list)))

            lambda_t = var(0.0)
            for i in lambda_list:
                lambda_t = lambda_t.add(i)
            lambda_t = lambda_t.div(var(len(lambda_list)))

            _, l_max = best_lambda(X_train, y_train, theta_t)
            _, l_min = best_theta(X_train, y_train, lambda_t)

            print('-------------------------------')
            print('l_max, l_min', l_max, l_min)

            if "gd" in optimizer_name:
                if (torch.abs(l_max.sub(l_min))).data.item() < w:
                # return theta_t, lambda_t
                    break
            else:
                if abs(l_max - l_min) < w:
                # return theta_t, lambda_t
                    break
            
            q = q.add(var(lr).mul(cal_c(X_train, y_train, theta)))

        eval(X_train, y_train, theta_t, target, 'train')
        eval(X_test, y_test, theta_t, target, 'test')

    # Eval
    # evaluation(X_train, y_train, theta_l, theta_r, target, lambda_=var(50.0), stop_val=stop_val, lr=lr)
    
    # # TEST
    # test(X_train, y_train, theta_l, theta_r, target)







