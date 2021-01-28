
import constants
from constants import *
from optimization import *
from test import *

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
    theta, loss, loss_list, q, c, time_out = optimize_f(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=stop_val, epoch=400, lr=lr)

    return theta, loss, time_out


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    for path_sample_size in sample_size_list:
        for data_for_sample in data_for_sample_list:
            time_out = False
            constants.SAMPLE_SIZE = path_sample_size # how many paths to sample?
            log_file = open(file_dir, 'a')
            log_file.write('####--PATH_SAMPLE_SIZE: ' + str(constants.SAMPLE_SIZE) + 'DATA_USED_FOR_SAMPLING_PATH:' + str(data_for_sample) + '\n')
            log_file.close()

            optimize_f = optimizer[optimizer_name]

            # data points generation
            target = domain.Interval(safe_l, safe_r)
            X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=data_size, target_theta=target_theta, test_size=test_portion)

            # add for lambdas
            # Loss(theta, lambda) = Q(theta) + lambda * C(theta)

            for i in range(5):
                lambda_list = list()
                theta_list = list()
                q = var(0.0)

                for t in range(t_epoch):
                    new_lambda = B.mul(q.exp().div(var(1.0).add(q.exp())))

                    # BEST_theta(lambda)
                    theta, loss, loss_list, q, c, time_out = optimize_f(X_train, y_train, theta_l, theta_r, target, lambda_=new_lambda, stop_val=stop_val, epoch=400, lr=lr)

                    #! To reduce time
                    theta_t = theta
                    break
                    
                    lambda_list.append(new_lambda)
                    theta_list.append(theta)

                    theta_t = list()
                    for i in range(len(theta)):
                        theta_t.append(var(0.0))

                    for i in theta_list:
                        # i is theta, is a list
                        for idx, value in enumerate(i):
                            theta_t[idx] = theta_t[idx].add(i)
                    for idx, value in enumerate(theta_t):
                        theta_t[idx] = theta_t[idx].div(var(len(theta_list)))

                    lambda_t = var(0.0)
                    for i in lambda_list:
                        lambda_t = lambda_t.add(i)
                    lambda_t = lambda_t.div(var(len(lambda_list)))

                    _, l_max = best_lambda(X_train, y_train, theta_t)
                    _, l_min, time_out = best_theta(X_train, y_train, lambda_t)

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
                
                if time_out == True:
                    break

                eval(X_train, y_train, theta_t, target, 'train')
                eval(X_test, y_test, theta_t, target, 'test')

            # Eval
            # evaluation(X_train, y_train, theta_l, theta_r, target, lambda_=var(50.0), stop_val=stop_val, lr=lr)

            # # TEST
            # test(X_train, y_train, theta_l, theta_r, target)







