import torch

def CUDA_wrapper(tensor):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor

class SoftmaxWithTemperature:
    def __init__(self, temperature):
        """
        formula: softmax(x/temperature)
        """
        self.temperature  = temperature
        self.softmax = torch.nn.Softmax()

    def __call__(self, x, temperature=None):
        if not temperature is None:
            return self.softmax(x / temperature)
        else:
            return self.softmax(x / self.temperature)

def fill_eye_diag(a):
    _, s1, s2 = a.data.shape
    dd = Variable(CUDA_wrapper(torch.eye(s1)))
    zero_dd = 1 - dd
    return a * zero_dd + dd
