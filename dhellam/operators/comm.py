import torch
from typing import Any, Dict, Union, Optional, Callable, Tuple

import torch.distributed
dist_group_type = torch.distributed.ProcessGroup

def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)

def gather_along_first_dim(
    input_: torch.Tensor, gpus:int,comm_group=None, async_op: bool = False,porfile:bool =False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather tensors and concatinate along the first dimension."""

    world_size = torch.distributed.get_world_size(comm_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=comm_group, async_op=async_op
    )

    return output, handle

def reduce_scatter_along_first_dim(
    input_: torch.Tensor,gpus:int,comm_group=None, async_op: bool = False,porfile:bool =False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = torch.distributed.get_world_size(comm_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    
    assert (
        dim_size[0] % world_size == 0
    ), f"First dimension :{dim_size[0]} of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=comm_group, async_op=async_op
    )
    return output, handle

def split_along_first_dim(input:torch.Tensor,gpus:int,tp_rank: int):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = gpus
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input

    # Split along first dimension.
    dim_size = input.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = tp_rank
    dim_offset = rank * local_dim_size

    output = input[dim_offset : dim_offset + local_dim_size].contiguous()

    return output

class ReqList:
    def __init__(self, reqlist: list) -> None:
        self.reqlist = reqlist
    def wait(self):
        for req in self.reqlist:
            req.wait()

def chunk_allgather_gemm(gemm_weight, comm_input, tp_size, tp_rank, tp_group):
    m1_dim0_size = comm_input.shape[0]*tp_size
    m2_dim1_size = gemm_weight.shape[1]
    result_mm = torch.empty(m1_dim0_size,m2_dim1_size,dtype=torch.float16,device='cuda')
    chunk_size = m1_dim0_size//tp_size
    result_buffer = [torch.empty_like(comm_input) if idx != tp_rank else comm_input for idx in range(tp_size)]
    tp_group_gid = torch.distributed.get_rank()//tp_size
    _TENSOR_PARALLEL_GLOBAL_RANKS = range(tp_group_gid*tp_size, (tp_group_gid+1)*tp_size)
    recv_src_rank = _TENSOR_PARALLEL_GLOBAL_RANKS[(tp_rank-1)%tp_size]
    send_dst_rank = _TENSOR_PARALLEL_GLOBAL_RANKS[(tp_rank+1)%tp_size]
    for idx in range(tp_size):
        reqs=None
        if idx<tp_size-1:
            send_data_idx = (idx + tp_rank)%tp_size
            recv_data_idx = (idx + tp_rank+1)%tp_size
            ops=[]
            send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend,
                    result_buffer[send_data_idx],
                    send_dst_rank,
                    tp_group,
                )
            ops.append(send_prev_op)
            recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    result_buffer[recv_data_idx],
                    recv_src_rank,
                    tp_group,
                )
            ops.append(recv_next_op)
            reqs = torch.distributed.batch_isend_irecv(ops)
        chunk_idx = (idx+tp_rank)%tp_size
        torch.matmul(result_buffer[chunk_idx],gemm_weight,out=result_mm[chunk_size*chunk_idx:(chunk_idx+1)*chunk_size])
        if reqs is not None:
            for req in reqs:
                req.wait()
    return torch.cat(result_buffer), result_mm

def chunk_gemm_reduce_scatter(gemm_input,gemm_weight,tp_size,tp_rank,tp_group):
    m1_dim0_size = gemm_input.shape[0]
    m2_dim1_size = gemm_weight.shape[1]
    result_mm = torch.empty(m1_dim0_size,m2_dim1_size,dtype=torch.float16,device='cuda')
    chunk_size = m2_dim1_size//tp_size
    result_buffer = []
    reqs = []
    for idx in range(tp_size):
        chunk_idx = (idx+tp_rank)%tp_size
        torch.matmul(gemm_input,gemm_weight[:,chunk_size*chunk_idx:(chunk_idx+1)*chunk_size],out=result_mm[:,chunk_size*chunk_idx:(chunk_idx+1)*chunk_size])
        torch.cuda.Stream.synchronize(torch.cuda.default_stream())
        comm_output,handle = reduce_scatter_along_first_dim(result_mm[:,chunk_size*chunk_idx:(chunk_idx+1)*chunk_size],0,tp_group,True)
        result_buffer.append(comm_output)
        reqs.append(handle)
    for req in reqs:
        req.wait()
    return torch.cat(result_buffer, dim=1)
def flash_attn_p2p_communicate(rank, send_tensor, send_dst,
                               recv_tensor, recv_src, cp_group, batch_p2p_comm=True):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []
    '''
    if tp_buffer is not None and tp_sync:
        if tp_buffer_list is None:
            torch.distributed.all_reduce(tp_buffer, op=torch.distributed.ReduceOp.SUM ,group=tp_group, async_op=False)
        else:
            torch.distributed.all_gather(tp_buffer_list,tp_buffer,group=tp_group,async_op=False)
    '''

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    '''
    if tp_buffer is not None and not tp_sync:
        if tp_buffer_list is None:
            tp_req = torch.distributed.all_reduce(tp_buffer, op=torch.distributed.ReduceOp.SUM ,group=tp_group, async_op=True)
            send_recv_reqs.append(tp_req)
        else:
            tp_req_1 = torch.distributed.all_gather(tp_buffer_list,tp_buffer,group=tp_group,async_op=True)
            send_recv_reqs.append(tp_req_1)
            tp_req_2 = torch.distributed.all_gather(tp_buffer_list,tp_buffer,group=tp_group,async_op=True)
            send_recv_reqs.append(tp_req_2)
    '''            

    return send_recv_reqs
