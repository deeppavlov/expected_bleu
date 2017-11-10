import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from copy import deepcopy
from collections import Counter
from copy import deepcopy as copy
from eBLEU import MultiBLEUMultiply
from eBLEU import CUDA_wrapper
import itertools
from functools import reduce

def one_hots(zeros, ix):
    for i in range(zeros.size()[0]):
        zeros[i, ix[i]] = 1
    return zeros

def O2(t, r, r_ids, f, temp):
    t_soft = f(t/temp)
    length = t.size()[0]
    v_size = t.size()[1]
    from_ref = [i.data[0] for i in r_ids]
    res = CUDA_wrapper(Variable(torch.Tensor([0])))
    for i in range(length - 1):
        p_0 = t_soft[i]
        p_1 = t_soft[i + 1]
        for m0 in from_ref:#range(v_size):
            for m1 in from_ref:#range(v_size):
                y_prod = (r[:-1, m0] * r[1:, m1]).sum(0)
                p_prod = t_soft[:-1, m0] * t_soft[1:, m1]
                # print(p_prod.data.shape)
                denominator = 1 + p_prod.sum(0) - p_prod[i]
                pr = p_0[m0] * p_1[m1]
                res += torch.min(pr, pr * y_prod / denominator)
    return res

def overlap(t, r_hot, r, f, temp, n):
    t_soft = f(t / temp)
    length = t.size()[0]
    v_size = t.size()[1]
    from_ref = [i.data[0] for i in r]
    res = CUDA_wrapper(Variable(torch.Tensor([0])))
    M = list(itertools.product(from_ref, repeat=n))
    mul = lambda x, y: x * y
    for i in range(length - n + 1):
        pp = [t_soft[i + j] for j in range(n)]
        for m in M:
            reslicer = lambda x: r.data.shape[0] if x == 0 else x
            y_prod = reduce(lambda x, y: (x * y).sum(0),\
                                   [r_hot[j:reslicer(-n + 1 + j), m[j]] for j in range(n)])
            if n == 1:
                y_prod = y_prod.sum(0)
            p_prod = reduce(mul, \
                                   [t_soft[j:reslicer(-n + 1 + j), m[j]] for j in range(n)])
            denominator = 1 + p_prod.sum(0) - p_prod[i]
            pr = reduce(mul, [pp[j][m[j]] for j in range(n)])

            res += torch.min(pr, pr * y_prod / denominator)
    return res

def precision(t, r_hot, r, f, temp, n):
    return overlap(t, r_hot, r, f, temp, n) / (t.data.shape[0] - n + 1)

def bleu(t, r_hot, r, f, temp, n):
    precisions = [precision(t, r_hot, r, f, temp, i) for i in range(1, n+1)]
    p_log_sum =  sum([(1. / n) * torch.log(p)\
                                                for p in precisions])
    return torch.exp(p_log_sum)
