import torch
import scipy.stats as ss
import math

from scipy.stats import truncnorm


def torch_normal_pdf(x, mean, std):
    x = torch.tensor(x)
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    var = std ** 2
    denom = (2*math.pi*var)**.5
    p = torch.exp(-((x-mean)**2)/(var*2))
    return p/denom


def normal_pdf(x, mean, std):
    var = std ** 2
    denom = (2*math.pi*var)**.5
    p = math.exp(-(x-mean)**2/(2*var))
    return p/denom


def scipy_norm_pdf(x, mean, std):
    p = ss.norm(mean, std).pdf(x)
    return p


def my_norm_pdf(x, mean, std):
    x = torch.tensor(x)
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    pi = torch.tensor(math.pi)
    p = torch.exp((-((x-mean)**2)/(2*std*std)))/ (std* torch.sqrt(2*pi))
    return p


def get_truncated_normal(mean, std, width):
    truncated_normal = truncnorm((mean-width-mean)/std, (mean+width-mean)/std, loc=mean, scale=std)
    return truncated_normal


if __name__ == "__main__":
    # mean = 0.3653
    # std = 0.1
    # x = 0.2690
    # print(f"mean: {mean}, std: {std}, x: {x}")

    # p1 = torch_normal_pdf(x, mean, std)
    # p2 = scipy_norm_pdf(x, mean, std)
    # p3 = normal_pdf(x, mean, std)
    # p4 = my_norm_pdf(x, mean, std)

    # print(f"torch_normal_pdf: {float(p1)}")
    # print(f"scipy_norm_pdf: {float(p2)}")
    # print(f"norm_pdf: {float(p3)}")s
    # print(f"my norm pdf: {float(p4)}")
    mean = torch.randn(2,3)
    std = 1.0
    width = 0.000001
    mean = mean.numpy()
    truncated_normal = get_truncated_normal(mean, std, width)
    print(mean)
    print(truncated_normal.rvs())


