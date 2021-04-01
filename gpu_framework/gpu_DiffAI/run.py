
from constants import *

from train import *
from evaluation import verification

from args import *

from data_generator import load_data
import random
import time


#TODO:  change arguments
def best_lambda(X_train, y_train, m, target):
    c = cal_c(X_train, y_train, m, target)
    q = cal_q(X_train, y_train, m)

    if c.data.item() <= 0.0:
        res_lambda = var(0.0)
    else:
        res_lambda = B
    return res_lambda, q.add(res_lambda.mul(c)) # lambda, L_max


# TODO: change arguments
def best_theta(component_list, lambda_, target):
    m, loss, loss_list, q, c, time_out = learning(
        component_list, 
        lambda_=lambda_, 
        stop_val=stop_val, 
        epoch=num_epoch, 
        lr=lr, 
        bs=bs,
        l=l,
        n=n,
        nn_mode=nn_mode,
        module=module,
        target=target,
        use_smooth_kernel=use_smooth_kernel,
        )

    return m, loss, time_out


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    for path_sample_size in path_num_list:

        time_out = False
        constants.SAMPLE_SIZE = path_sample_size # show number of paths to sample
        log_file = open(file_dir, 'a')
        log_file.write(f"path_sample_size: {path_sample_size}\n")
        log_file.close()

        target = {
            "condition": domain.Interval(var(SAFE_RANGE[0]), var(SAFE_RANGE[1])),
            "phi": var(PHI),
        }

        # data points generation
        preprocessing_time = time.time()
        # TODO: the data is one-dimension (x = a value)
        X_train, X_test, y_train, y_test = load_data(train_size=train_size, test_size=test_size, dataset_path=DATASET_PATH)
        component_list = extract_abstract_representation(X_train, y_train, x_l, x_r, num_components)
        print(f"prepare data: {time.time() - preprocessing_time}")
        # Loss(theta, lambda) = Q(theta) + lambda * C(theta)

        for i in range(5):
            lambda_list = list()
            model_list = list()
            q = var(0.0)

            for t in range(t_epoch):
                new_lambda = B.mul(q.exp().div(var(1.0).add(q.exp())))

                # BEST_theta(lambda)
                m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
                if test_mode:
                    epochs_to_skip, m = load_model(m, MODEL_PATH, name=f"{benchmark_name}_{data_attr}_{n}_{lr}_{use_smooth_kernel}")
                    # TODO: for quick result
                    if m is not None and epochs_to_skip is not None:
                        print(f"Load Model.")
                        break
                else:
                    epochs_to_skip = None

                m, loss, loss_list, q, c, time_out = learning(
                    m, 
                    component_list,
                    lambda_=new_lambda, 
                    stop_val=stop_val, 
                    epoch=num_epoch, 
                    target=target,
                    lr=lr, 
                    bs=bs,
                    n=n,
                    nn_mode=nn_mode,
                    l=l,
                    module=module,
                    use_smooth_kernel=use_smooth_kernel, 
                    save=save,
                    epochs_to_skip=epochs_to_skip,
                    )
                m.eval()

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

                # TODO: change
                _, l_max = best_lambda(X_train, y_train, m_t, target)
                _, l_min, time_out = best_theta(X, Y, abstract_representation, lambda_t, target)

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

            # TODO: add verification and test
            # verification, going through the program without sampling
            # test for the quantitative accuracy

            print(f"------------start verification------------")
            verification_time = time.time()
            verification(model_path=MODEL_PATH, model_name=f"{benchmark_name}_{data_attr}_{n}_{lr}_{use_smooth_kernel}", component_list=component_list, target=target)
            print(f"---verification time: {time.time() - verification_time} sec---")
            exit(0)
            # eval(X_train, y_train, m_t, target, 'train')
            # eval(X_test, y_test, m_t, target, 'test')








