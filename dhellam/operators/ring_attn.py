import torch
from importlib.metadata import version
from pkg_resources import packaging
import torch.distributed
from transformer_engine.pytorch.jit import jit_fuser, no_torch_dynamo
from dhellam.operators.comm import flash_attn_p2p_communicate
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
)

_flash_attn_version = packaging.version.Version(version("flash-attn"))
_flash_attn_version_required = packaging.version.Version("2.0.6")
_flash_attn_2_1_plus = _flash_attn_version >= packaging.version.Version("2.1")
_flash_attn_2_3_plus = _flash_attn_version >= packaging.version.Version("2.3")
_flash_attn_2_4_plus = _flash_attn_version >= packaging.version.Version("2.4")
_flash_attn_2_4_1_plus = _flash_attn_version >= packaging.version.Version("2.4.1")

if _flash_attn_version >= _flash_attn_version_required:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_forward_func # pylint: disable=no-name-in-module
    from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd # pylint: disable=no-name-in-module
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_forward # pylint: disable=no-name-in-module,ungrouped-imports
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_backward # pylint: disable=no-name-in-module


@jit_fuser
def flash_attn_fwd_softmax_lse_correction(softmax_lse, softmax_lse_per_step):
    """Merge softmax stats of each step in Attention with context parallelism"""
    softmax_lse.exp_()
    softmax_lse.add_(softmax_lse_per_step.to(torch.double).exp())
    softmax_lse.log_()


@jit_fuser
def flash_attn_fwd_out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).transpose(1, 2)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step*softmax_lse_corrected_exp
    out.add_(out_corrected)


def fa_cp_block_fwd(q,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,p2p_comm_buffers,out_per_step, softmax_lse_per_step,rng_states,softmax_lse,softmax_lse_, i,causal,cp_group,cp_global_ranks,stream):

    '''
    if causal:    
        # [b, s, np, hn] -> [b, 2, s//2, np, hn]
        q, k, v = [x.view(x.shape[0], 2, x.shape[1]//2, *x.shape[2:]) for x in [q, k, v]]
    '''
     
    assert(q.shape[-1] % 8 == 0), "hidden size per attention head should be multiple of 8"

    fa_optional_forward_kwargs = {}
    if _flash_attn_2_3_plus:
        fa_optional_forward_kwargs["window_size"] = [-1, 0] if causal else [-1, -1]
    if _flash_attn_2_4_plus:
        fa_optional_forward_kwargs["alibi_slopes"] = None

    q_input = None
    rank = get_distributed_rank(cp_group)
    cp_size = get_distributed_world_size(cp_group)
    kv_inputs = p2p_comm_buffers[i]

    if i < cp_size-1 :
        
        send_dst = cp_global_ranks[(rank + 1) % cp_size]
        recv_src = cp_global_ranks[(rank + cp_size - 1) % cp_size]
        
        if i == 0:
            out_per_step = [None for _ in range(cp_size)]
            softmax_lse_per_step = [None for _ in range(cp_size)]
            rng_states = [None for _ in range(cp_size)]
            # p2p_comm_buffers = [None for _ in range(cp_size)]

        p2p_comm_buffers[i+1] = torch.empty_like(p2p_comm_buffers[i])
        p2p_reqs = flash_attn_p2p_communicate(
                                    rank,
                                    kv_inputs,
                                    send_dst,
                                    p2p_comm_buffers[i+1],
                                    recv_src,
                                    cp_group,
                                    batch_p2p_comm=True
                                            )

    if stream is None:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        if causal:
            if i == 0:
                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                q_input = q.view(-1, *q.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                kv_inputs = kv_inputs.view(2, -1, *q.shape[-2:])
                _, _, _, _, out_per_step[i], \
                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                    q_input, kv_inputs[0], kv_inputs[1],
                    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    0.1, 0.125, causal=True, return_softmax=False,
                    **fa_optional_forward_kwargs
                )
            elif i <= rank:
                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                q_input = q.view(-1, *q.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk//2, np, hn]
                kv_inputs = kv_inputs[:, :, 0, ...].contiguous()
                # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                kv_inputs = kv_inputs.view(2, -1, *q.shape[-2:])
                if _flash_attn_2_3_plus:
                    fa_optional_forward_kwargs["window_size"] = [-1, -1]
                _, _, _, _, out_per_step[i], \
                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                    q_input, kv_inputs[0], kv_inputs[1],
                    cu_seqlens_q, cu_seqlens_k//2, max_seqlen_q, max_seqlen_k//2,
                    0.1, 0.125, causal=False, return_softmax=False,
                    **fa_optional_forward_kwargs
                )
            else:  
                # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                q_input = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                kv_inputs = kv_inputs.view(2, -1, *q.shape[-2:])
                if _flash_attn_2_3_plus:
                    fa_optional_forward_kwargs["window_size"] = [-1, -1]
                _, _, _, _, out_per_step[i], \
                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                    q_input, kv_inputs[0], kv_inputs[1],
                    cu_seqlens_q//2, cu_seqlens_k, max_seqlen_q//2, max_seqlen_k,
                    0.1, 0.125, causal=False, return_softmax=False,
                    **fa_optional_forward_kwargs
                )
        else:
            # [b, sq, np, hn] -> [b*sq, np, hn]
            q_input = q.view(-1, *q.shape[-2:])
            # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
            kv_inputs = kv_inputs.view(2, -1, *q.shape[-2:])
            _, _, _, _, out_per_step[i], \
            softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                q_input, kv_inputs[0], kv_inputs[1],
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                0.1, 0.125, causal=False, return_softmax=False,
                **fa_optional_forward_kwargs
            )

        if i > 0:  #TODO 
            # wait until fwd restuls correction of last step is done
            '''
            if i > 1:
                flash_attn_streams[(i-1)%2].wait_event(fwd_results_correction_done)
            '''
            if i == 1:
                softmax_lse = torch.clone(softmax_lse_per_step[0]).to(torch.double)
                if causal:
                    # [b, np, sq] -> [b, np, 2, sq//2]
                    softmax_lse_ = softmax_lse.view(
                        *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1]//2
                    )
            elif (i-1) <= rank or not causal:
                flash_attn_fwd_softmax_lse_correction(softmax_lse,
                                                        softmax_lse_per_step[i-1])
            else:
                flash_attn_fwd_softmax_lse_correction(softmax_lse_[..., 1, :],
                                                        softmax_lse_per_step[i-1])

            '''
            if i < cp_size:
                flash_attn_streams[(i-1)%2].record_event(fwd_results_correction_done)
            '''

        if i== cp_size-1:
            if i <= rank or not causal:
                flash_attn_fwd_softmax_lse_correction(softmax_lse,
                                                        softmax_lse_per_step[i])
            else:
                flash_attn_fwd_softmax_lse_correction(softmax_lse_[..., 1, :],
                                                        softmax_lse_per_step[i])

            out = torch.empty_like(q).zero_()
            softmax_lse = softmax_lse.to(torch.float)

            
            for i in range(cp_size):
                # [b*sq, np, hn] -> [b, sq, np, hn] or [b*sq//2, np, hn] -> [b, sq//2, np, hn]
                out_ = out_per_step[i].view(out.shape[0], -1, *out.shape[-2:])
                if i <= rank or not causal:
                    flash_attn_fwd_out_correction(out.view(*out_.shape),
                                                    out_,
                                                    softmax_lse,
                                                    softmax_lse_per_step[i])
                else:
                    flash_attn_fwd_out_correction(out[:, 1, ...],
                                                    out_,
                                                    softmax_lse_[..., 1, :],
                                                    softmax_lse_per_step[i])
            
            out = out.view(-1, *out.shape[-2:])
            kv = p2p_comm_buffers[-1]
            return out,q,kv,softmax_lse,rng_states

        return [q,p2p_comm_buffers,out_per_step,softmax_lse_per_step,rng_states,softmax_lse,softmax_lse_,i+1],p2p_reqs
    

def fa_cp_block_bwd(dout,out,q,dq,kv,dkv_,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,p2p_comm_buffers,i,causal,cp_group,cp_global_ranks,stream):

    deterministic = False

    cp_size = get_distributed_world_size(cp_group)
    rank = get_distributed_rank(cp_group)
    send_dst = cp_global_ranks[(rank + cp_size - 1) % cp_size]
    recv_src = cp_global_ranks[(rank + 1) % cp_size]


    if causal:
        softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, softmax_lse.shape[-1]//2)
        softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
    
    out = out.view(*q.shape)
    dout = dout.view(*q.shape)

    fa_optional_backward_kwargs = {}
    if _flash_attn_2_4_plus:
        fa_optional_backward_kwargs["alibi_slopes"] = None
    

    if _flash_attn_2_4_1_plus:
        fa_optional_backward_kwargs["deterministic"] = deterministic  # default value is False

    '''
    # wait until KV is received
    for req in send_recv_reqs:
        req.wait()
    '''

    with torch.cuda.stream(stream):

        if i > 0 :

            dkv = p2p_comm_buffers[i%2][1]
            '''
            if ctx.use_fused_attention:
                dkv_ = torch.cat((dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0)
            '''
            if causal and i-1 >= (cp_size-rank-1) and i-1 != (cp_size-1):
                # [2, b*sk//2, np, hn] -> [2, b, sk//2, np, hn]
                dkv_ = dkv_.view(*dkv.shape[0:2], *dkv.shape[3:])
            else:
                # [2, b*sk, np, hn] -> [2, b, 2, sk//2, np, hn] if causal
                # [2, b*sk, np, hn] -> [2, b, sk, np, hn] if not causal
                dkv_ = dkv_.view(*dkv.shape)

            if causal:
                if i-1 == (cp_size-1):
                    if rank == 0:
                        dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                        dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                    else:
                        dkv.add_(dkv_)
                elif i-1 >= (cp_size-rank-1):
                    if i-1 == 0 and rank == (cp_size-1):
                        dkv[:, :, 0, ...].copy_(dkv_)
                    else:
                        dkv[:, :, 0, ...].add_(dkv_)
                elif i-1 > 0:
                    dkv.add_(dkv_)
                else:
                    dkv.copy_(dkv_)
            else:
                if i-1 == 0:
                    dkv.copy_(dkv_)
                else:
                    dkv.add_(dkv_)
            
            if i == cp_size :
                if causal:
                    # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                    dq = dq.view(q.shape[0], -1, *q.shape[-2:])
                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                    dkv = dkv.view(*kv.shape[0:2], -1, *kv.shape[-2:])
                
                    return  dq, dkv

        send_tensor = p2p_comm_buffers[i%2]
        recv_tensor = p2p_comm_buffers[(i+1)%2]
        if i == 0:
            send_tensor = send_tensor[0]
            recv_tensor = recv_tensor[0]
        if i == (cp_size-1):
            send_tensor = send_tensor[1]
            recv_tensor = recv_tensor[1]

        p2p_reqs = flash_attn_p2p_communicate(rank,
                                                    send_tensor,
                                                    send_dst,
                                                    recv_tensor,
                                                    recv_src,
                                                    cp_group,
                                                    batch_p2p_comm=True)

        kv = p2p_comm_buffers[i%2][0]
        # In reversed order of fwd
        if causal:
            if i == (cp_size-1):
                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                q_ = q.view(-1, *q.shape[-2:])
                dq_ = torch.empty_like(q_)
                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                kv_ = kv.view(2, -1, *kv.shape[-2:])
                dkv_ = torch.empty_like(kv_)
                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                out_ = out.view(-1, *out.shape[-2:])
                dout_ = dout.view(-1, *dout.shape[-2:])
                if _flash_attn_2_3_plus:
                    fa_optional_backward_kwargs["window_size"] = [-1, 0]
                _flash_attn_backward(
                    dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                    dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k,
                    0.1, 0.125, True,
                    rng_state= rng_states[cp_size-i-1],
                    **fa_optional_backward_kwargs
                )
            elif i >= (cp_size-rank-1):
                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                q_ = q.view(-1, *q.shape[-2:])
                dq_ = torch.empty_like(q_)
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                kv_ = kv[:, :, 0, ...].contiguous().view(2, -1, *kv.shape[-2:])
                dkv_ = torch.empty_like(kv_)
                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                out_ = out.view(-1, *out.shape[-2:])
                dout_ = dout.view(-1, *dout.shape[-2:])
                if _flash_attn_2_3_plus:
                    fa_optional_backward_kwargs["window_size"] = [-1, -1]
                _flash_attn_backward(
                    dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                    dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k//2,
                    max_seqlen_q, max_seqlen_k//2,
                    0.1, 0.125, False,
                    rng_state= rng_states[cp_size-i-1],
                    **fa_optional_backward_kwargs
                )
            else:
                # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                q_ = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                dq_ = torch.empty_like(q_)
                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                kv_ = kv.view(2, -1, *kv.shape[-2:])
                dkv_ = torch.empty_like(kv_)
                # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                out_ = out[:, 1, ...].contiguous().view(-1, *out.shape[-2:])
                dout_ = dout[:, 1, ...].contiguous().view(-1, *dout.shape[-2:])
                if _flash_attn_2_3_plus:
                    fa_optional_backward_kwargs["window_size"] = [-1, -1]
                _flash_attn_backward(
                    dout_, q_, kv_[0], kv_[1], out_, softmax_lse_,
                    dq_, dkv_[0], dkv_[1], cu_seqlens_q//2, cu_seqlens_k,
                    max_seqlen_q//2, max_seqlen_k,
                    0.1, 0.125, False,
                    rng_state = rng_states[cp_size-i-1],
                    **fa_optional_backward_kwargs
                )
        else:
            # [b, sq, np, hn] -> [b*sq, np, hn]
            q_ = q.view(-1, *q.shape[-2:])
            dq_ = torch.empty_like(q_)
            # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
            kv_ = kv.view(2, -1, *kv.shape[-2:])
            dkv_ = torch.empty_like(kv_)
            # [b, sq, np, hn] -> [b*sq, np, hn]
            out_ = out.view(-1, *out.shape[-2:])
            dout_ = dout.view(-1, *dout.shape[-2:])
            if _flash_attn_2_3_plus:
                fa_optional_backward_kwargs["window_size"] = [-1, -1]
            _flash_attn_backward(
                dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                0.1, 0.125, False,
                **fa_optional_backward_kwargs
            )

        if i >= (cp_size-rank-1) or not causal:
            # [b*sq, np, hn] -> [b, 2, sq//2, np, hn] if causal
            # [b*sq, np, hn] -> [b, sq, np, hn] if not causal
            dq_ = dq_.view(*dq.shape)
        else:
            # [b*sq//2, np, hn] -> [b, sq//2, np, hn]
            dq_ = dq_.view(dq.shape[0], *dq.shape[2:])

        if causal:
            if i > (cp_size-rank-1):
                dq.add_(dq_)
            elif i == (cp_size-rank-1):
                if rank == (cp_size-1):
                    dq.copy_(dq_)
                else:
                    dq[:, 0, ...].copy_(dq_[:, 0, ...])
                    dq[:, 1, ...].add_(dq_[:, 1, ...])
            elif i > 0:
                dq[:, 1, ...].add_(dq_)
            else:
                dq[:, 1, ...].copy_(dq_)
        else:
            if i == 0:
                dq.copy_(dq_)
            else:
                dq.add_(dq_)

    
    return [dout,dq,dkv_,p2p_comm_buffers,i+1],p2p_reqs
    





