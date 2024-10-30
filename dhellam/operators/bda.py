import torch

@torch.compile
def bda_fwd_(input:torch.Tensor, residual):
    mask = torch.nn.functional.dropout(torch.ones_like(input, dtype=torch.int8), 0.1, True, True)
    input.mul_(mask.half()).mul_(0.9).add_(residual)
    return input, mask

@torch.compile
def bda_bwd_(dgrad, mask, output):
    output.add_(mask.half()).mul_(dgrad).mul_(0.9)
    return dgrad, output

def bda_fwd(input, residual, stream=None, profiling=False): 
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            output, mask = bda_fwd_(input, residual)
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end)
        else:
            output, mask = bda_fwd_(input, residual)
            elapsed_time = 0
            
    return output, mask, elapsed_time

def bda_bwd(dgrad, mask, output=None, stream=None, profiling=False): 
    if stream is None:
        stream = torch.cuda.current_stream()
    if output is None:
        output = torch.zeros_like(dgrad)
    with torch.cuda.stream(stream):
        if profiling:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            dgrad_residual, dgrad_input = bda_bwd_(dgrad, mask, output)
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end)
        else:
            dgrad_residual, dgrad_input = bda_bwd_(dgrad, mask, output)
            elapsed_time = 0
    return dgrad_residual, dgrad_input, elapsed_time
    