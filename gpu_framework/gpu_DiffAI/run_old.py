
from constants import *

from train_dse import *
from test import *

from args import *
import random

optimizer = {
    'gd_direct_noise': gd_direct_noise,
    # 'direct': direct,
    # 'gd_gaussian_noise': gd_gaussian_noise,
    # 'gd': gd,
}


def best_lambda(X_train, y_train, m, target):
    c = cal_c(X_train, y_train, m, target)
    q = cal_q(X_train, y_train, m)

    if c.data.item() <= 0.0:
        res_lambda = var(0.0)
    else:
        res_lambda = B
    return res_lambda, q.add(res_lambda.mul(c)) # lambda, L_max


def best_theta(X_train, y_train, lambda_, target):
    m, loss, loss_list, q, c, time_out = optimize_f(
        X_train, 
        y_train, 
        theta_l, 
        theta_r, 
        target, 
        lambda_=lambda_, 
        stop_val=stop_val, 
        epoch=num_epoch, 
        lr=lr, 
        bs=bs,
        l=l,
        n=n,
        nn_mode=nn_mode,
        module=module,
        )

    return m, loss, time_out


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    for path_sample_size in sample_size_list:
        for safe_idx, safe_l in enumerate(sliced_safe_l_list):
            safe_r = sliced_safe_r_list[safe_idx]

            time_out = False
            constants.SAMPLE_SIZE = path_sample_size # how many paths to sample?
            log_file = open(file_dir, 'a')
            log_file.write(f"####--PATH_SAMPLE_SIZE: {constants.SAMPLE_SIZE}, safe_range: [{safe_l}, {safe_r}]\n")
            log_file.close()

            optimize_f = optimizer[optimizer_name]

            # data points generation
            target = domain.Interval(var(safe_l), var(safe_r))
            X_train, X_test, y_train, y_test = data_generator(x_l, x_r, size=data_size, target_theta=target_theta, test_size=test_portion)

            # add for lambdas
            # Loss(theta, lambda) = Q(theta) + lambda * C(theta)

            for i in range(5):
                lambda_list = list()
                model_list = list()
                q = var(0.0)

                for t in range(t_epoch):
                    new_lambda = B.mul(q.exp().div(var(1.0).add(q.exp())))

                    # BEST_theta(lambda)
                    m, loss, loss_list, q, c, time_out = optimize_f(
                        X_train, 
                        y_train, 
                        theta_l, 
                        theta_r, 
                        target, 
                        lambda_=new_lambda, 
                        stop_val=stop_val, 
                        epoch=num_epoch, 
                        lr=lr, 
                        bs=bs,
                        n=n,
                        nn_mode=nn_mode,
                        l=l,
                        module=module)

                    #TODO: reduce time, because there are some issues with the gap between cal_c and cal_q
                    m_t = m
                    break
                    
                    lambda_list.append(new_lambda)
                    model_list.append(model)

                    # TODO: return a distribution
                    m_t = random.choice(model_list)

                    lambda_t = var(0.0)
                    for i in lambda_list:
                        lambda_t = lambda_t.add(i)
                    lambda_t = lambda_t.div(var(len(lambda_list)))

                    _, l_max = best_lambda(X_train, y_train, m_t, target)
                    _, l_min, time_out = best_theta(X_train, y_train, lambda_t, target)

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
                    
                    q = q.add(var(lr).mul(cal_c(X_train, y_train, m_t, theta)))
                
                if time_out == True:
                    break

                eval(X_train, y_train, m_t, target, 'train')
                eval(X_test, y_test, m_t, target, 'test')








