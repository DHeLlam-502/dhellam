from dhellam.adaptor.megatron import initialize_dhellam_megatron
import os
args = initialize_dhellam_megatron(os.getenv('MegaPath',None))
from dhellam.operators.comm import gather_along_first_dim, reduce_scatter_along_first_dim, chunk_allgather_gemm, chunk_gemm_reduce_scatter
from dhellam.common.benchmark import benchmark_ms
from dhellam.common.common import print_rank0
import torch
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_data_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from dhellam.operators.layernorm import pylayernorm_fwd, pylayernorm_bwd
from dhellam.operators.swiglu import swiglu, swiglu_back
from dhellam._Clib import pymha_varlen_fwd, pymha_varlen_bwd
from dhellam.operators.bda import bda_fwd, bda_bwd

repeat = 10
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

def bda_bwd_(args):
    input = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, args.hidden_size, dtype=torch.float16,device="cuda")
    mask = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, args.hidden_size, dtype=torch.float16,device="cuda")
    return [
        input,
        mask
    ], bda_bwd

def bda_(args):
    input = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, args.hidden_size,dtype=torch.float16,device="cuda")
    residual = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, args.hidden_size,dtype=torch.float16,device="cuda")
    return [
        input,
        residual
    ], bda_fwd

def fa_bwd(args):
    q =    torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    k =    torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    v =    torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    dout = torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    out =  torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    cu_seqlens_k = torch.tensor(list(range(0,args.micro_batch_size*args.seq_length+1,args.seq_length)), device='cuda', dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,args.micro_batch_size*args.seq_length+1,args.seq_length)), device='cuda', dtype=torch.int32)
    softmax_lse = torch.randn(args.micro_batch_size, args.num_attention_heads//args.tensor_model_parallel_size, args.seq_length, dtype=torch.float32,device='cuda')
    rng_state = torch.tensor([813829799034691,           72172], device='cuda')
    return [
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        args.seq_length,
        args.seq_length,
        0.1,
        0.01,
        True,
        rng_state,
        False,
        0,
        False,
        torch.cuda.current_stream().cuda_stream
    ], pymha_varlen_bwd

def fa_(args):
    q = torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    k = torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    v = torch.randn(args.micro_batch_size*args.seq_length, args.num_attention_heads//args.tensor_model_parallel_size, args.hidden_size//args.num_attention_heads, dtype=torch.float16,device='cuda')
    cu_seqlens_k = torch.tensor(list(range(0,args.micro_batch_size*args.seq_length+1,args.seq_length)), device='cuda', dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,args.micro_batch_size*args.seq_length+1,args.seq_length)), device='cuda', dtype=torch.int32)
    return [
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        args.seq_length,
        args.seq_length,
        0.1,
        0.125,
        True,
        False,
        0,
        False], pymha_varlen_fwd

def swiglu_bwd(args):
    dgrad = torch.randn(args.micro_batch_size*args.seq_length,2*args.hidden_size//args.tensor_model_parallel_size,dtype=torch.float16,device="cuda")
    input = torch.randn(args.micro_batch_size*args.seq_length,4*args.hidden_size//args.tensor_model_parallel_size,dtype=torch.float16,device="cuda")

    return [
        dgrad,
        input
    ], swiglu_back

def swiglu_(args):
    input = torch.randn(args.micro_batch_size*args.seq_length,4*args.hidden_size//args.tensor_model_parallel_size,dtype=torch.float16,device="cuda")
    return [input,], swiglu

def ln_bwd(args):
    input = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size,args.hidden_size, dtype=torch.float16, device='cuda')
    doutput= torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size,args.hidden_size, dtype=torch.float16, device='cuda')
    weight = torch.randn(args.hidden_size, dtype=torch.float16,device="cuda")
    bias =torch.randn(args.hidden_size, dtype=torch.float16,device="cuda")
    mu= torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, dtype=torch.float32,device="cuda")
    rsigma= torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, dtype=torch.float32,device="cuda")
    return [
        doutput,
        input,
        mu,
        rsigma,
        weight,
        0,
        False,
        None,
        False
    ], pylayernorm_bwd

def ln_(args):
    input = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size,args.hidden_size,dtype=torch.float16,device="cuda")
    weight = torch.randn(args.hidden_size,dtype=torch.float16,device="cuda")
    bias = torch.randn(args.hidden_size,dtype=torch.float16,device="cuda")
    input_args = [input,weight,bias,0.9,0,False,None,False]
    return input_args, pylayernorm_fwd

def gemm(args):
    gemm_l = torch.randn(args.micro_batch_size*args.seq_length, args.hidden_size, dtype=torch.float16, device='cuda')
    gemm_r = torch.randn(args.hidden_size, 2*args.ffn_hidden_size//args.tensor_model_parallel_size, dtype=torch.float16, device='cuda')
    return [gemm_l,gemm_r], torch.matmul

def rs(args):
    rs_input = torch.randn(args.micro_batch_size*args.seq_length//args.tensor_model_parallel_size, args.hidden_size, dtype=torch.float16, device='cuda')
    return [rs_input, 0, get_tensor_model_parallel_group(),True], gather_along_first_dim

def a2a(args):
    a2a_input = torch.randn(args.micro_batch_size*args.seq_length//2, args.hidden_size, dtype=torch.float16, device='cuda')
    a2a_output = torch.empty_like(a2a_input)
    torch.distributed.all_to_all_single(a2a_output,a2a_input,group=get_data_parallel_group())
    return [a2a_output, a2a_input, None, None, get_data_parallel_group(),True], torch.distributed.all_to_all_single

def empty_func(args):
    def empty_kernel():
        return
    return [],empty_kernel

def synchronize():
    torch.distributed.barrier()
    torch.cuda.synchronize()

def overlap_eff(a,b,c):
    return (a+b-c)/min(a,b)

def run_test(args, fn1_str, fn2_str, s1, s2):
    fn1 = op_table_fwd[fn1_str]
    fn2 = op_table_bwd[fn2_str]
    # warmup
    benchmark_ms(             3,s1,s2,fn1,fn2,args)
    synchronize()
    fn1_time =  benchmark_ms(10,s1,s1,fn1,empty_func,args)
    synchronize()
    fn2_time =  benchmark_ms(10,s1,s1,fn2,empty_func,args)
    synchronize()
    para_time = benchmark_ms(10,s1,s2,fn1,fn2,args)
    synchronize()

    speedup = (fn1_time+fn2_time)/para_time
    eff = overlap_eff(fn1_time, fn2_time, para_time)
    print_rank0(f"[{fn1_str} - {fn2_str}] speedup: {speedup:.2f} eff: {eff:.2f}")
    return speedup, eff

op_table_fwd = {
    "gemm": gemm,
    "bda": bda_,
    "ln": ln_,
    "swiglu": swiglu_,
    "fa": fa_,
    "rs":rs,
    "a2a": a2a
}
op_table_bwd = {
    "gemm": gemm,
    "bda_bwd": bda_bwd_,
    "ln_bwd": ln_bwd,
    "swiglu_bwd": swiglu_bwd,
    "fa_bwd": fa_bwd,
    "rs": rs,
    "a2a": a2a
}

keys_fwd = list(op_table_fwd.keys())
keys_bwd = list(op_table_bwd.keys())

speedup_result=list()
eff_result=list()

for i in keys_fwd:
    tmp_speedup_list = []
    tmp_eff_list = []
    for j in keys_bwd:
        speedup, eff = run_test(args,i,j,stream1,stream2)
        tmp_speedup_list.append(speedup)
        tmp_eff_list.append(eff)
    speedup_result.append(tmp_speedup_list)
    eff_result.append(tmp_eff_list)

if torch.distributed.get_rank()==0:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = pd.DataFrame(speedup_result,index=keys_fwd,columns=keys_bwd)
    csv_file_path = "speedup.csv"
    df.to_csv(csv_file_path, index=False)

    df = pd.DataFrame(eff_result,index=keys_fwd,columns=keys_bwd)
    csv_file_path = "efficiency.csv"
    df.to_csv(csv_file_path, index=False)





