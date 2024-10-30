import torch


def template_fwd(fwd_func,inputs,stream,profiling):
    elapsed_time=0
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs= fwd_func(inputs)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            outputs= fwd_func(inputs)
    return [inputs,outputs],elapsed_time

def template_bwd(bwd_func,ctx,grad_outputs,stream,profiling):
    elapsed_time=0
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            grads= bwd_func(ctx,grad_outputs)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            grads = bwd_func(ctx,grad_outputs)
    return grads, elapsed_time

def template_fwd_new(fwd_func,*inputs,stream=None,profiling=False):
    elapsed_time=0
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs= fwd_func(*inputs)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            outputs= fwd_func(*inputs)
    return [[each for each in inputs if isinstance(each, torch.Tensor) and each.requires_grad],outputs],elapsed_time

def template_bwd_new(ctx,grad_outputs,stream=None,profiling=False):
    elapsed_time=0
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            torch.autograd.backward(ctx[1],grad_tensors=grad_outputs,inputs=ctx[0] if ctx[0] else None)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            torch.autograd.backward(ctx[1],grad_tensors=grad_outputs,inputs=ctx[0] if ctx[0] else None)
    return [each.grad for each in ctx[0]], elapsed_time