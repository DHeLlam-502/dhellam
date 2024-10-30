import torch
from typing import Optional, Tuple, Union
from dhellam._Clib import pygemm as gemm
from dhellam.common.common import get_workspace,print_rank0
from torch import matmul
import time
def gemm_fwd(
            args,
            input:torch.Tensor,
            weight:torch.Tensor,
            output: Optional[torch.Tensor] = None,
            layout: str = "TN",
            profiling:bool = False,
            math_sm_count:int = 0,
            is_matmul:bool = False,
            stream:torch.Stream = None
            ):
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        if is_matmul:
            if transa:
                weight = torch.transpose(weight, dim0=0, dim1=1)
            if transb:
                input = torch.transpose(input,dim0=0,dim1=1)
            if profiling:
                stream.synchronize()
                start = time.time()
            
            output = matmul(input,weight)
            if profiling:
                stream.synchronize()
                gemm_time = (time.time()-start)*1000
            else:
                gemm_time = 0 
        else:
            if output is None:
                output = torch.empty(
                    input.shape[1] if transb else input.shape[0],
                    weight.shape[0] if transa else weight.shape[1],
                    dtype=torch.float16,
                    device=f"cuda:{torch.cuda.current_device()}",
                )
            work_tensor = get_workspace()
            #weight first input*weight = [bs,hd]*[qkv,hd]^T
            gemm_args = (
                weight,
                input,
                output,
                work_tensor,
                work_tensor[0],
                transa,
                transb,
                False,
                math_sm_count,
                profiling
            )
            fn = gemm
            gemm_time = fn(*gemm_args)
    return output,gemm_time

def gemm_fwd_event(
            args,
            input:torch.Tensor,
            weight:torch.Tensor,
            output: Optional[torch.Tensor] = None,
            layout: str = "TN",
            profiling:bool = False,
            math_sm_count:int = 0,
            is_matmul:bool = False
            ):
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    if is_matmul:
        if transa:
            weight = torch.transpose(weight, dim0=0, dim1=1)
        if transb:
            input = torch.transpose(input,dim0=0,dim1=1)
        if profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start = time.time()
        
        output = matmul(input,weight)
        if profiling:
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            gemm_time = elapsed_time
        else:
            gemm_time = 0 
    else:
        if output is None:
            output = torch.empty(
                input.shape[1] if transb else input.shape[0],
                weight.shape[0] if transa else weight.shape[1],
                dtype=torch.float16,
                device=f"cuda:{torch.cuda.current_device()}",
            )
        work_tensor = get_workspace()
        #weight first input*weight = [bs,hd]*[qkv,hd]^T
        gemm_args = (
            weight,
            input,
            output,
            work_tensor,
            work_tensor[0],
            transa,
            transb,
            False,
            math_sm_count,
            profiling
        )
        fn = gemm
        gemm_time = fn(*gemm_args)
    return output,gemm_time

def gemm_bwd(
        args,
        input_grad:torch.Tensor,
        weight:torch.Tensor,
        input_data:torch.Tensor,
        dgrad_output:torch.Tensor,
        wgrad_output:torch.Tensor,
        grad_accumulate: bool = False,
        layout: str = "NNNT",
        profiling:bool = False,
        math_sm_count:int = 0,
        is_matmul:bool = False):
    dgrad_transa = layout[0] == "T"
    dgrad_transb = layout[1] == "T"
    wgrad_transa = layout[2] == "T"
    wgrad_transb = layout[3] == "T"
    #input grad [bs,qkv]
    if is_matmul:
        if dgrad_transa:
            weight = torch.transpose(weight, dim0=0, dim1=1)
        if dgrad_transb:
            input_grad = torch.transpose(input_grad,dim0=0,dim1=1)
        current_stream = torch.cuda.current_stream()
        current_stream.synchronize()
        start = time.time()
        dgrad_output = matmul(input_grad,weight)
        current_stream.synchronize()
        dgrad_time = (time.time()-start)*1000

        if wgrad_transa:
            input_data = torch.transpose(input_data, dim0=0, dim1=1)
        if wgrad_transb:
            input_grad = torch.transpose(input_grad,dim0=0,dim1=1)
        current_stream = torch.cuda.current_stream()
        current_stream.synchronize()
        start = time.time()
        wgrad_output = matmul(input_grad,input_data)
        current_stream.synchronize()
        wgrad_time = (time.time()-start)*1000

    else:
        if dgrad_output is None:
            dgrad_output = torch.empty(
                input_grad.shape[1] if dgrad_transb else input_grad.shape[0],
                weight.shape[0] if dgrad_transa else weight.shape[1],
                dtype=torch.float16,
                device=f"cuda:{torch.cuda.current_device()}",
            )
        work_tensor = get_workspace()
        #weight first input_grad*weight [bs,qkv]*[qkv,hd] dgrad don't need accumulate
        dgrad_args = (
            weight,
            input_grad,
            dgrad_output,
            work_tensor,
            work_tensor[0],
            dgrad_transa,
            dgrad_transb,
            False,
            math_sm_count,
            profiling
        )
        fn = gemm
        dgrad_time = fn(*dgrad_args)

        if wgrad_output is None:
            wgrad_output = torch.empty(
                input_grad.shape[1] if wgrad_transb else input_grad.shape[0],
                input_data.shape[0] if wgrad_transa else input_data.shape[1],
                dtype=torch.float16,
                device=f"cuda:{torch.cuda.current_device()}",
            )
        #weight first input_grad^T =input_data[bs,qkv]^T*[bs,hidden] maybe need accumulate
        wgrad_args = (
            input_data,
            input_grad,
            wgrad_output,
            work_tensor,
            work_tensor[0],
            wgrad_transa,
            wgrad_transb,
            grad_accumulate,
            math_sm_count,
            profiling
        )
        wgrad_time = fn(*wgrad_args)
    return dgrad_output,wgrad_output,dgrad_time,wgrad_time

def dgrad_bwd(
        args,
        input_grad:torch.Tensor,
        weight:torch.Tensor,
        dgrad_output:torch.Tensor,
        layout: str = "NN",
        profiling:bool = False,
        math_sm_count:int = 0,
        is_matmul:bool = False,
        stream:torch.Stream = None):
    dgrad_transa = layout[0] == "T"
    dgrad_transb = layout[1] == "T"
    if stream is None:
        stream = torch.cuda.current_stream()
    if dgrad_output is None:
        dgrad_output = torch.empty(
            input_grad.shape[1] if dgrad_transb else input_grad.shape[0],
            weight.shape[0] if dgrad_transa else weight.shape[1],
            dtype=torch.float16,
            device=f"cuda:{torch.cuda.current_device()}",
        )
    with torch.cuda.stream(stream):
        if is_matmul:
            if dgrad_transa:
                weight = torch.transpose(weight, dim0=0, dim1=1)
            if dgrad_transb:
                input_grad = torch.transpose(input_grad,dim0=0,dim1=1)
            if profiling:
                stream.synchronize()
                start = time.time()
            matmul(input_grad,weight,out=dgrad_output)
            if profiling:    
                stream.synchronize()
                dgrad_time = (time.time()-start)*1000
            else:
                dgrad_time = 0
        else:
            #dgrad [bs,qkv]
            work_tensor = get_workspace()
            #weight first input_grad*weight [bs,qkv]*[qkv,hd] dgrad don't need accumulate
            dgrad_args = (
                weight,
                input_grad,
                dgrad_output,
                work_tensor,
                work_tensor[0],
                dgrad_transa,
                dgrad_transb,
                False,
                math_sm_count,
                profiling
            )
            fn = gemm
            dgrad_time = fn(*dgrad_args)
    return dgrad_output,dgrad_time

def wgrad_bwd(
        args,
        input_grad:torch.Tensor,
        input_data:torch.Tensor,
        wgrad_output:torch.Tensor,
        grad_accumulate: bool = False,
        layout: str = "NT",
        profiling:bool = False,
        math_sm_count:int = 0,
        is_matmul:bool = False,
        stream:torch.Stream = None):
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        wgrad_transa = layout[0] == "T"
        wgrad_transb = layout[1] == "T"
        if is_matmul:
            if wgrad_transa:
                input_data = torch.transpose(input_data, dim0=0, dim1=1)
            if wgrad_transb:
                input_grad = torch.transpose(input_grad,dim0=0,dim1=1)
            if profiling:
                stream.synchronize()
                start = time.time()
            if wgrad_output is None:
                wgrad_output = matmul(input_grad,input_data)
            else:
                wgrad_output = torch.addmm(wgrad_output, input_grad,input_data,out=wgrad_output)
            if profiling:
                stream.synchronize()
                wgrad_time = (time.time()-start)*1000
            else:
                wgrad_time = 0
        else:
            # wgrad_output [qkv,hd] input_grad [bs,qkv]  
            if wgrad_output is None:
                wgrad_output = torch.empty(
                    input_grad.shape[1] if wgrad_transb else input_grad.shape[0],
                    input_data.shape[0] if wgrad_transa else input_data.shape[1],
                    dtype=torch.float16,
                    device=f"cuda:{torch.cuda.current_device()}",
                )
            work_tensor = get_workspace()
            #weight first input_grad^T =input_data[bs,qkv]^T*[bs,hidden] maybe need accumulate
            wgrad_args = (
                input_data,
                input_grad,
                wgrad_output,
                work_tensor,
                work_tensor[0],
                wgrad_transa,
                wgrad_transb,
                grad_accumulate,
                math_sm_count,
                profiling
            )
            fn = gemm
            wgrad_time = fn(*wgrad_args)
    return wgrad_output,wgrad_time