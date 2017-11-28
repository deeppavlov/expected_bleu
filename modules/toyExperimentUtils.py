import torch
from torch.autograd import Variable
# from modules.expectedBLEU import bleu as ebleu
from modules.expectedBLEU import log_bleu as ebleu
from modules.utils import CUDA_wrapper
import numpy as np
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt

def one_hots(size, ix):
    zeros = torch.zeros(size)
    for i in range(zeros.size()[0]):
        zeros[i, ix[i]] = 1
    return zeros

def training(t, r_hot, r, f, n, mbleu):
    res = []
    bleus = []
    norms = []
    probs = []
    opt = torch.optim.Adam([t], lr=10)
    gradients = []
    for i in range(100):
        b2 = ebleu(t, r_hot, r, f, 1, n)
        res.append(b2.data[0])
        probs.append(f(t).data.cpu().numpy())
        (-b2).backward()
        # print("gradient of eblu")
        # print(t.grad)
        gradients.append(t.grad)
        opt.step()
        norms.append(t.grad.data.norm())
        hard_t = Variable(CUDA_wrapper(\
                        one_hots(list(t.size()), torch.max(t, dim=1)[1].data)))
        bleus.append(-mbleu(torch.unsqueeze(r_hot, 0),
                            torch.unsqueeze(hard_t, 0), [10], [10])[0].data[0])
    return res, bleus, probs, gradients

def estimate_expectation(r_hot, probs, mbleu, length, n_exp = 100):
    bs = []
    bst = []
    for p in probs:
        bs = []
        for i in range(n_exp):
            samples = []
            for l in range(length):
                samples.append(np.random.multinomial(1, \
                                            p[l]/ np.sum(p[l]) - 1E-7))
            text = np.array(samples).astype(np.float32)
            bs.append(-mbleu(torch.unsqueeze(r_hot, 0),
                            torch.unsqueeze(\
                            Variable(CUDA_wrapper(torch.from_numpy(text))), 0),
                            [10], [10])[0].data[0])
        bst.append(np.array(bs).mean())
    return bst


def plot_results(bleus, bst, res):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}
    set_matplotlib_formats('png')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(bleus, '-', linewidth=2, label='Exact BLEU with argmax')
    plt.plot(bst, 'x', linewidth=6, label='Expected BLEU')
    plt.plot(res, '--', linewidth=2, label='Lower Bound for Expected BLEU')
    plt.ylabel('BLEU score')
    plt.xlabel('training steps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
