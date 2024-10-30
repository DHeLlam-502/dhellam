import transformer_engine
import transformer_engine_extensions as tex
import torch
from typing import Union, Tuple, Optional
import time
def pylayernorm_fwd(inp: torch.Tensor,
                    ln_weight: torch.Tensor,
                    ln_bias: torch.Tensor,
                    eps: float,
                    fwd_sm_margin: int,
                    zero_centered_gamma: bool,
                    stream: torch.cuda.Stream,
                    profiling: bool=False,
                    ) -> torch.Tensor:
    
    in_features = ln_weight.numel()
    assert inp.is_cuda, "dhellam needs CUDA."
    assert inp.shape[-1] == in_features, "LayerNorm not possible"
    inputmat = inp.view((-1, in_features))
    elapsed_time=0

    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            # ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight,
            #     ln_bias, eps, fwd_sm_margin, zero_centered_gamma)
            # end_event.record()
            # end_event.synchronize()
            # elapsed_time = start_event.elapsed_time(end_event)
            current_stream =  torch.cuda.current_stream()
            current_stream.synchronize()
            start_time = time.time()
            ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight,
                ln_bias, eps, fwd_sm_margin, zero_centered_gamma)
            current_stream.synchronize()
            elapsed_time = time.time()-start_time
        else:
            ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight,
                ln_bias, eps, fwd_sm_margin, zero_centered_gamma)
    
    return ln_out.view_as(inp),mu,rsigma,elapsed_time

def pylayernorm_bwd(
                    d_ln_out: torch.Tensor,
                    inp: torch.Tensor,
                    mu: torch.Tensor,
                    rsigma: torch.Tensor,
                    ln_weight: torch.Tensor,
                    bwd_sm_margin: int,
                    zero_centered_gamma: bool,
                    stream: torch.cuda.Stream,
                    profiling: bool=False,
                    ) -> Tuple[Union[torch.Tensor, None,float], ...]:
    
    in_features = ln_weight.numel()
    assert inp.is_cuda, "TransformerEngine needs CUDA."
    assert inp.shape[-1] == in_features, "LayerNorm not possible"
    inputmat = inp.view((-1, in_features))
    d_ln_out = d_ln_out.view((-1, in_features))
    elapsed_time=0

    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if profiling:
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            # dinp, dgamma, dbeta = tex.layernorm_bwd(
            # d_ln_out, inputmat, mu, rsigma, ln_weight,
            # bwd_sm_margin,zero_centered_gamma)
            # end_event.record()
            # end_event.synchronize()
            # elapsed_time = start_event.elapsed_time(end_event)
            current_stream =  torch.cuda.current_stream()
            current_stream.synchronize()
            start_time = time.time()
            dinp, dgamma, dbeta = tex.layernorm_bwd(d_ln_out, inputmat, mu, rsigma, ln_weight,
            bwd_sm_margin,zero_centered_gamma)
            current_stream.synchronize()
            elapsed_time = time.time()-start_time
        else:
            dinp, dgamma, dbeta = tex.layernorm_bwd(
            d_ln_out, inputmat, mu, rsigma, ln_weight,
            bwd_sm_margin,zero_centered_gamma)

    return dinp.view_as(inp), dgamma, dbeta,elapsed_time
