import torch

def f(t):
    return t[:, :, 1::2]

dummy_inputs = torch.rand(2, 2, 5)
m = torch.jit.trace(f, dummy_inputs)
m.save('test.jit')
