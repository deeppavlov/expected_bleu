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
from modules.matrixBLEU import mBLEU
from modules.utils import CUDA_wrapper
import itertools
from functools import reduce
from modules.utils import LongTensor, FloatTensor
import time

def one_hots(zeros, ix):
    for i in range(zeros.size()[0]):
        zeros[i, ix[i]] = 1
    return zeros

def overlap(t, r_hot, r, f, temp, n):
    """ calculate overlap as in original BLEU script but expected.
    see google's nmt bleu.py BLEU script for details """
    t_soft = f(t / temp)
    length = t.size()[0]
    v_size = t.size()[1]
    from_ref = list([i.data[0] for i in r])
    from_ref_t = LongTensor(from_ref)
    mapper_ref = {j:i for i, j in enumerate(from_ref)}
    res = CUDA_wrapper(Variable(FloatTensor([0])))
    M = [[from_ref[i + j] for j in range(n)] for i in range(len(from_ref) - n + 1)]
    mul = lambda x, y: x * y
    start_all = time.time()
    cum_selecting_time = 0
    cum_iterating_time = 0
    for i in range(length - n + 1):
        start_select_t_soft = time.time()
        pp = [t_soft[i + j] for j in range(n)]
        # print('-' * 10 + 'selecting_time' + '-' * 10)
        now_selecting_time = time.time() - start_select_t_soft
        # print('selecting_time: {0:0.4f}'.format(now_selecting_time))
        cum_selecting_time += now_selecting_time
        start_iterating = time.time()
        # print("length all m combos: {0:0.4f}".format(len(M)))
        ngram_calc_cum = 0
        for m in M:
            reslicer = lambda x: r.data.shape[0] + x
            ngram_calc_start = time.time()
            y_prod = reduce(mul,
                     [r_hot[j:reslicer(-n + 1 + j), m[j]] for j in range(n)]) # j is id of current word in sentense
            y_prod = y_prod.sum(0)
            p_prod = reduce(mul, \
                     [t_soft[j:reslicer(-n + 1 + j), m[j]] for j in range(n)])
            denominator = 1 + p_prod.sum(0) - p_prod[i]
            ngram_calc_cum += time.time() - ngram_calc_start
            pr = reduce(mul, [pp[j][m[j]] for j in range(n)])
            res += torch.min(pr, pr * y_prod / denominator)
        now_iterating_time = time.time() - start_iterating
        # print('iterating_time: {0:0.4f}'.format(now_iterating_time))
        # print('ngram_calc_cum: {0:0.4f}'.format(ngram_calc_cum))
        cum_iterating_time += now_iterating_time
    # print('cum_selecting_time: {0:0.4f}'.format(cum_selecting_time))
    # print('cum_iterating_time: {0:0.4f}'.format(cum_iterating_time))
    # print('-'*20)
    return res

def precision(t, r_hot, r, f, temp, n):
    return overlap(t, r_hot, r, f, temp, n) / (t.data.shape[0] - n + 1)

def bleu(t, r_hot, r, f, temp, n):
    precisions = [precision(t, r_hot, r, f, temp, i) for i in range(1, n+1)]
    p_log_sum =  sum([(1. / n) * torch.log(p)\
                                                for p in precisions])
    return torch.exp(p_log_sum)
