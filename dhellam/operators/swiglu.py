import torch
import torch.nn.functional as F

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
jit_fuser = torch.jit.script

# nvFuser is deprecated in PyTorch JIT starting from 2.2
if (TORCH_MAJOR > 2) or (TORCH_MAJOR == 2 and TORCH_MINOR >= 2):
    jit_fuser = torch.compile

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################
@jit_fuser
def swiglu_(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back_(g, y, out):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )

def swiglu(y, stream=None, profiling=False):
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            result = swiglu_(y)
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end)
        else:
            result = swiglu_(y)
            elapsed_time = 0
    return result, elapsed_time

def swiglu_back(g, y, o=None, stream=None, profiling=False):
    if stream is None:
        stream = torch.cuda.current_stream()
    if o is None:
        o = torch.empty_like(y)
    with torch.cuda.stream(stream):
        if profiling:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            o= swiglu_back_(g, y, o)
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end)
        else:
            o = swiglu_back_(g, y, o)
            elapsed_time = 0
    return o, elapsed_time
