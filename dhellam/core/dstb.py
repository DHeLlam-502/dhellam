import torch
from torch.nn.parameter import Parameter
from enum import Enum
import json
from dhellam import gemm_fwd,pylayernorm_fwd,gather_along_first_dim,pymha_varlen_fwd,_get_qkv_layout,reduce_scatter_along_first_dim,gemm_fwd_event
from dhellam.operators.bda import bda_fwd,bda_bwd
from dhellam.operators.swiglu import swiglu,swiglu_back
from dhellam.common.common import print_rank0
from dhellam import gemm_bwd,pylayernorm_bwd,gather_along_first_dim,pymha_varlen_bwd,_get_qkv_layout,reduce_scatter_along_first_dim
from dhellam.operators.layernorm_column import dgrad_bwd,wgrad_bwd
from megatron import get_args
from cofutils import cofnsys,cofcsv
from typing import List, Type
from dhellam.common.global_variables import GLOBAL_VARIABLES
from dhellam.operators.ring_attn import fa_cp_block_fwd,fa_cp_block_bwd

MOVABLE_OPERATOR=[
]

def check_available(tensor):
    if tensor is not None:
        return torch.isfinite(tensor).all()
    else:
        return True
class FuncChoice(Enum):
    ln_attn = 1
    ln_mlp = 2
    gemm_attn_column = 3
    gemm_attn_linear = 4
    gemm_mlp_column =5
    gemm_mlp_linear = 6
    attn = 7
    bda_mlp = 8
    bda_attn = 9
    reduce_scatter = 10
    all_gather = 11
    swiglu = 12
    gemm_dgrad_mlp_c = 13
    gemm_wgrad_mlp_c = 14
    gemm_dgrad_mlp_l = 15
    gemm_wgrad_mlp_l = 16
    gemm_dgrad_attn_c = 17
    gemm_wgrad_attn_c = 18
    gemm_dgrad_attn_l = 19
    gemm_wgrad_attn_l = 20
    gemm_wgrad_mlp_c_n = 114
    gemm_wgrad_mlp_l_n = 116
    gemm_wgrad_attn_c_n = 118
    gemm_wgrad_attn_l_n = 120
    swiglu_bwd  = 21
    bda_mlp_bwd = 22
    bda_bwd_attn= 23
    attn_bwd   = 24
    ln_bwd_mlp = 25
    ln_bwd_attn= 26
    allgather_ln_mlp = 27
    allgather_ln_attn = 28
    ring_attn_fwd = 29
    ring_attn_bwd = 30

def bytes_to_gb(bytes):
    """将字节转换为吉字节（GB）"""
    return bytes / (1024**3)

class DCFunc:
    @staticmethod
    def ln_attn(config, data, stream):
        input, ln_weight, ln_bias = data
        in_features = ln_weight.numel()
        inputmat = input.view((-1, in_features))
        output, attn_mu, attn_rsigma,_ = pylayernorm_fwd(inputmat,ln_weight,ln_bias,1e-05,0,False,stream,profiling=False)
        return output, attn_mu, attn_rsigma, inputmat, input, output
    @staticmethod
    def ln_mlp(config, data, stream):
        input, ln_weight, ln_bias = data
        in_features = ln_weight.numel()
        inputmat = input.view((-1, in_features))
        output, mlp_mu, mlp_rsigma,_ = pylayernorm_fwd(inputmat,ln_weight,ln_bias,1e-05,0,False,stream,profiling=False)
        return output,mlp_mu, mlp_rsigma,inputmat,input,output
    
    @staticmethod
    def gemm_attn_column(config, data, stream): 
        #*  应该在这里特判一下应该就行，为CP>1的时候搞一个特殊的场景
        input, attn_column_weight = data
        output,_ = gemm_fwd(config,input,attn_column_weight,None,"TN",False,0,True,stream)

        if config.context_parallel_size > 1:
            qkv = output
            output = [qkv,None,None,None,None,None,None,0] # for attn in CP
        
        return output
    @staticmethod
    def gemm_attn_linear(config, data, stream):
        input, attn_linear_weight = data
        input = input.view(-1,input.shape[-1])
        output,_ = gemm_fwd(config,input,attn_linear_weight,None,"TN",False,0,True,stream)
        return output, input
    @staticmethod
    def gemm_mlp_column(config, data, stream):
        input, mlp_column_weight = data
        output,_ = gemm_fwd(config,input,mlp_column_weight,None,"TN",False,0,True,stream)
        return output
    @staticmethod
    def gemm_mlp_linear(config, data, stream):
        input, mlp_linear_weight = data
        input = input.view(config.micro_batch_size*config.seq_length,-1)
        output,_ = gemm_fwd(config,input,mlp_linear_weight,None,"TN",False,0,True,stream)
        return output, input
    @staticmethod
    def attn(config, data, stream):
        qkv, cu_seqlens_q, cu_seqlens_kv = data
        new_tensor_shape = qkv.size()[:-1] + (
            config.num_attention_heads//config.tensor_model_parallel_size,
            (config.hidden_size//config.num_attention_heads)*3)
        #[bs,3*h//tp]->[bs,nh//tp,3*h//nh]
        mixed_qkv = qkv.view(*new_tensor_shape)
        split_arg_list = [
            config.hidden_size//config.num_attention_heads,
            config.hidden_size//config.num_attention_heads,
            config.hidden_size//config.num_attention_heads
        ]
        #[bs,nh//tp,3*h//nh]->3*[bs, nh//tp, hn] hn==head hidden
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=2,)
        with torch.cuda.stream(stream):
            q, k, v = [x.contiguous()
                    for x in (query, key, value)]
            attn_output, q_padded,k_padded,v_padded,out_padded,softmax_lse, p, rng_state = pymha_varlen_fwd(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        config.seq_length,
                        config.seq_length,
                        0.1,
                        0.125,
                        True,
                        False,
                        0,
                        False
                    )
            output = attn_output.view(config.micro_batch_size,config.seq_length, -1).contiguous()
            
        return output,q_padded,k_padded,v_padded,out_padded,rng_state,softmax_lse 

    @staticmethod
    def ring_attn_fwd(config,data,stream):
        input,cu_seqlens_q, cu_seqlens_kv = data
        qkv,p2p_comm_buffers,out_per_step,softmax_lse_per_step,rng_states,softmax_lse,softmax_lse_,cp_iter_times = input

        causal =  True #* 对于decoder only模型来说应该都是true
        cp_group = config.cp_group
        cp_size = config.context_parallel_size
        tp_size =  config.tp_size
        cp_global_ranks = config.cp_global_ranks
        #q,k,v,cu_seqlens_q,cu_seqlens_kv,causal,cp_size,p2p_comm_buffers,out_per_step,softmax_lse_per_step,rng_states,count_times = data

        if p2p_comm_buffers is None:
            #TODO check the correctness of matrix split
            qkv_reshape =  qkv.view(config.micro_batch_size, config.seq_length, config.num_attention_heads//tp_size, 3*(config.hidden_size//config.num_attention_heads))
            split_arg_list = [
                config.hidden_size//config.num_attention_heads,
                config.hidden_size//config.num_attention_heads,
                config.hidden_size//config.num_attention_heads
            ]
            q, k, v = torch.split(qkv_reshape, split_arg_list, dim=3,)
            q, k, v = [x.contiguous() for x in [q, k, v]]
            if causal:
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [x.view(x.shape[0], 2, x.shape[1]//2, *x.shape[2:]) for x in [q, k, v]]
            # kv_inputs = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
            p2p_comm_buffers = [None for _ in range(cp_size)]
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
            qkv = q

        output = fa_cp_block_fwd(qkv,
                                cu_seqlens_q=cu_seqlens_q,
                                cu_seqlens_k=cu_seqlens_kv,
                                max_seqlen_q=config.seq_length,
                                max_seqlen_k=config.seq_length,
                                p2p_comm_buffers=p2p_comm_buffers,
                                out_per_step=out_per_step,
                                softmax_lse_per_step=softmax_lse_per_step,
                                rng_states=rng_states,
                                softmax_lse= softmax_lse,
                                softmax_lse_= softmax_lse_,
                                i= cp_iter_times,
                                causal= causal,
                                cp_group= cp_group,
                                cp_global_ranks= cp_global_ranks,
                                stream=stream)
        
        if cp_iter_times == cp_size-1:
            out,q,kv,softmax_lse,rng_states = output
            out = out.view(config.micro_batch_size,config.seq_length, -1).contiguous()
            return out,q,kv,softmax_lse,rng_states

        return output
    
    @staticmethod
    def ring_attn_bwd(config,data,stream):
        input,out,q,kv,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_kv = data
        dout,dq,dkv_,p2p_comm_buffers,cp_iter_times = input
        
        '''
        if cp_iter_times ==1:
            if dist.get_rank() ==0:
                print("[jqruan] Tensor Shape: ",out.shape,"Tensor dtype: ",out.dtype)
                print("[jqruan] Tensor Shape: ",q.shape,"Tensor dtype: ",q.dtype)
                print("[jqruan] Tensor Shape: ",kv.shape,"Tensor dtype: ",kv.dtype)
                print("[jqruan] Tensor Shape: ",softmax_lse.shape,"Tensor dtype: ",softmax_lse.dtype)
                print("[jqruan] Tensor Shape: ",rng_states[0].shape,"Tensor dtype: ",rng_states[0].dtype)
                print("[jqruan] Tensor Shape: ",dout.shape,"Tensor dtype: ",dout.dtype)
                print("[jqruan] Tensor Shape: ",dq.shape,"Tensor dtype: ",dq.dtype)
                print("[jqruan] Tensor Shape: ",dkv_.shape,"Tensor dtype: ",dkv_.dtype)
        '''

        causal = True
        cp_group = config.cp_group
        cp_global_ranks = config.cp_global_ranks

        if p2p_comm_buffers is None:
            #TODO check the correctness
            p2p_comm_buffers = [torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device), \
                                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)]
            dq = torch.empty_like(q)
            p2p_comm_buffers[0][0].copy_(kv)
        
        output = fa_cp_block_bwd(dout,
                                 out=out,
                                 q=q,
                                 dq=dq,
                                 kv=kv,
                                 dkv_= dkv_,
                                 softmax_lse=softmax_lse,
                                 rng_states=rng_states,
                                 cu_seqlens_q=cu_seqlens_q,
                                 cu_seqlens_k=cu_seqlens_kv,
                                 max_seqlen_q=config.seq_length,
                                 max_seqlen_k=config.seq_length,
                                 p2p_comm_buffers=p2p_comm_buffers,
                                 i=cp_iter_times,
                                 causal=causal,
                                 cp_group=cp_group,
                                 cp_global_ranks=cp_global_ranks,
                                 stream=stream)
        
        if cp_iter_times == config.context_parallel_size:
            #TODO correctness check
            with torch.cuda.stream(stream):
                dq,dkv = output
                dk = dkv[0]
                dv = dkv[1]
                dq = dq.view(-1, *dq.shape[-2:])
                dk = dk.view(-1, *dk.shape[-2:])
                dv = dv.view(-1, *dv.shape[-2:])

                output = torch.empty(dq.size(0),dq.size(1),dq.size(2) + dk.size(2) + dv.size(2), dtype=torch.float16,device=torch.cuda.current_device())
                #[bs,hn//tp,3hn]
                torch.cat((dq, dk, dv), dim=2, out=output)
                #[bs,3h//tp]
                output = output.view(output.shape[0],-1)

        return output
    

    @staticmethod
    def bda_mlp(config, data, stream):
        input, residual = data
        input = input.view(config.micro_batch_size,config.seq_length//config.tensor_model_parallel_size,-1)
        output,mlp_mask,_ = bda_fwd(input,residual,stream,False)
        return output,mlp_mask
    @staticmethod
    def bda_attn(config, data, stream):
        input, residual = data
        input = input.view(config.micro_batch_size,config.seq_length//config.tensor_model_parallel_size,-1)
        output,attn_mask,_ = bda_fwd(input,residual,stream,False)
        return output,attn_mask
    @staticmethod
    def swiglu(config, data, stream):
        input, = data
        input = input.view(config.micro_batch_size,config.seq_length,-1)
        output,_ = swiglu(input,stream,False)
        return output
    @staticmethod
    def bda_attn_bwd(config, data, stream):
        grad_output, mask = data
        grad_output = grad_output.view(config.micro_batch_size,-1,grad_output.shape[-1])
        attn_residual_bwd, output, _ = bda_bwd(grad_output,mask,None,stream,False)
        output = output.view(-1,output.shape[-1]) 
        return output, attn_residual_bwd
    @staticmethod
    def bda_mlp_bwd(config, data, stream):
        grad_output, mask = data
        mlp_residual_bwd, output, _ = bda_bwd(grad_output,mask,None,stream,False)
        output = output.view(-1,output.shape[-1])
        return output, mlp_residual_bwd
    @staticmethod
    def gemm_dgrad_mlp_c(config, data, stream):
        grad_output, mlp_column_weight = data
        output,_ = dgrad_bwd(config,grad_output,mlp_column_weight,None,is_matmul=True,stream=stream)
        return output
    @staticmethod
    def gemm_dgrad_mlp_l(config, data, stream):
        grad_output, mlp_linear_weight = data
        output,_ = dgrad_bwd(config,grad_output,mlp_linear_weight,None,is_matmul=True,stream=stream)
        return output
    @staticmethod
    def gemm_dgrad_attn_c(config, data, stream):
        grad_output,attn_column_weight = data
        output,_ = dgrad_bwd(config,grad_output,attn_column_weight,None,is_matmul=True,stream=stream)
        return output
    @staticmethod
    def gemm_dgrad_attn_l(config,data,stream):
        grad_output,attn_linear_weight = data
        output,_ = dgrad_bwd(config,grad_output,attn_linear_weight,None,is_matmul=True,stream=stream)
        #TODO 在这里要针对CP>1进行更改
        if config.context_parallel_size >1:
            dout = output
            output = [dout,None,None,None,0]
        return output
    @staticmethod
    def gemm_wgrad_attn_c(config,data,stream):
        input, grad_output, weight_main_grad = data
        wgrad,_ = wgrad_bwd(config,grad_output,input,weight_main_grad,is_matmul=True,stream=stream)
        # print_rank0(f"weight_main_grad:{weight_main_grad.size()} wgrad:{wgrad.size()}")
        return None
    @staticmethod
    def gemm_wgrad_attn_l(config,data,stream):
        grad_output, input, weight_main_grad = data
        wgrad_bwd(config,grad_output,input,weight_main_grad,is_matmul=True,stream=stream)
        return None
    @staticmethod
    def gemm_wgrad_mlp_c(config,data,stream):
        input, grad_output, weight_main_grad = data
        wgrad_bwd(config,grad_output,input,weight_main_grad,is_matmul=True,stream=stream)
        return None
    @staticmethod
    def gemm_wgrad_mlp_l(config,data,stream):
        grad_output, input, weight_main_grad = data
        wgrad_bwd(config,grad_output,input,weight_main_grad,is_matmul=True,stream=stream)
        return None
    @staticmethod
    def swiglu_bwd(config,data,stream):
        # [bs,*]
        swiglu_input, grad_output = data
        swiglu_input = swiglu_input.view(-1,swiglu_input.shape[-1])
        swiglu_dgrad,_=swiglu_back(grad_output,swiglu_input,None,stream,profiling=False)
        return swiglu_dgrad
    @staticmethod
    def attn_bwd(config,data,stream):
        grad_output,q_padded,k_padded,v_padded,out_padded,softmax_lse,cu_seqlens_q,cu_seqlens_kv,rng_state = data
        new_tensor_shape = grad_output.size()[:-1] + (
            config.num_attention_heads//config.tensor_model_parallel_size,
            (config.hidden_size//config.num_attention_heads))
        #[bs,h//tp] ->[bs,nh//tp,hn]
        grad_output = grad_output.view(*new_tensor_shape).contiguous()
        dq,dk,dv,*_= pymha_varlen_bwd(grad_output
                                    ,q_padded
                                    ,k_padded
                                    ,v_padded
                                    ,out_padded
                                    ,softmax_lse
                                    ,cu_seqlens_q
                                    ,cu_seqlens_kv
                                    ,config.seq_length
                                    ,config.seq_length
                                    ,0.1
                                    ,0.125
                                    ,True
                                    ,rng_state
                                    ,False
                                    ,0
                                    ,False
                                    ,stream.cuda_stream)
        
        dq = dq[..., : grad_output.shape[-1]]
        dk = dk[..., : grad_output.shape[-1]]
        dv = dv[..., : grad_output.shape[-1]]
        dgrad_output = torch.empty(dq.size(0),dq.size(1),dq.size(2) + dk.size(2) + dv.size(2), dtype=torch.float16,device=torch.cuda.current_device())
        with torch.cuda.stream(stream):
            #[bs,hn//tp,3hn]
            torch.cat((dq, dk, dv), dim=2, out=dgrad_output)
        #[bs,3h//tp]
        dgrad_output = dgrad_output.view(dgrad_output.shape[0],-1)
        return dgrad_output
    @staticmethod
    def ln_mlp_bwd(config,data,stream):
        output, ln_input, ln_mu, ln_rsigma, ln_weight, ln_bias, residual, ln_weight_main_grad, ln_bias_main_grad = data
        dgrad, mlp_dgamma, mlp_dbeta,_ = pylayernorm_bwd(output,ln_input,ln_mu,ln_rsigma,ln_weight,0,False,stream,False)
        residual = residual.view(-1,residual.shape[-1])
        with torch.cuda.stream(stream):
            dgrad.add_(residual)
            ln_weight_main_grad.add_(mlp_dgamma)
            ln_bias_main_grad.add_(mlp_dbeta)
        return dgrad
    @staticmethod
    def ln_attn_bwd(config,data,stream):
        output, ln_input, ln_mu, ln_rsigma, ln_weight, ln_bias, residual, ln_weight_main_grad, ln_bias_main_grad = data
        dgrad, attn_dgamma, attn_dbeta,_ = pylayernorm_bwd(output,ln_input,ln_mu,ln_rsigma,ln_weight,0,False,stream,False)
        dgrad = dgrad.view(config.micro_batch_size,-1,residual.shape[-1])
        with torch.cuda.stream(stream):
            dgrad.add_(residual)
            ln_weight_main_grad.add_(attn_dgamma)
            ln_bias_main_grad.add_(attn_dbeta)
        return dgrad
    @staticmethod
    def reduce_scatter(config,data,stream):
        input,async_op = data
        out, handle = reduce_scatter_along_first_dim(input,gpus=config.tensor_model_parallel_size,comm_group=config.tp_group,async_op=async_op)
        return out,handle
    @staticmethod
    def allgather(config,data,stream):
        input,async_op = data
        out,handle = gather_along_first_dim(input,gpus=config.tensor_model_parallel_size,comm_group=config.tp_group,async_op=async_op)
        return out,handle
    @staticmethod
    def allgather_ln_mlp(config,data,stream):
        input,async_op = data
        out,handle = gather_along_first_dim(input,gpus=config.tensor_model_parallel_size,comm_group=config.tp_group,async_op=async_op)
        return out,handle
    @staticmethod
    def allgather_ln_attn(config,data,stream):
        input,async_op = data
        out,handle = gather_along_first_dim(input,gpus=config.tensor_model_parallel_size,comm_group=config.tp_group,async_op=async_op)
        return out,handle
    
class DCoperation:
    @staticmethod
    def perform_operation(dcfunc,operation,config,weight,activation_dict,**kwargs):
        stream = kwargs['stream']
        if operation == 1:
            input, ln_weight, ln_bias = kwargs['output'], weight["ln_attn_weight"], weight["ln_attn_bias"]
            output\
            ,activation_dict["attn_mu"]\
            ,activation_dict["attn_rsigma"]\
            ,activation_dict["ln_attn_input"]\
            ,activation_dict["ln_attn_residual"]\
            ,_ = dcfunc.ln_attn(config, (input, ln_weight, ln_bias), stream)
        if operation == 2:
            input, ln_weight, ln_bias = kwargs['output'], weight["ln_mlp_weight"], weight["ln_mlp_bias"]
            output\
            ,activation_dict["mlp_mu"]\
            ,activation_dict["mlp_rsigma"]\
            ,activation_dict["ln_mlp_input"]\
            ,activation_dict["ln_mlp_residual"]\
            ,_ = dcfunc.ln_mlp(config, (input, ln_weight, ln_bias), stream)
        if operation == 3:
            input, attn_column_weight = kwargs['output'], weight["column_attn_weight"]
            activation_dict["gemm_attn_c_input"] = input
            output = dcfunc.gemm_attn_column(config,(input, attn_column_weight),stream)
        if operation == 4:
            input, attn_linear_weight = kwargs['output'], weight["linear_attn_weight"]
            output, activation_dict["linear_attn_input"] = dcfunc.gemm_attn_linear(config,(input, attn_linear_weight),stream)
        if operation == 5:
            input, mlp_column_weight = kwargs['output'], weight["column_mlp_weight"]
            activation_dict["gemm_mlp_c_input"] = input
            output = dcfunc.gemm_mlp_column(config,(input, mlp_column_weight),stream)
        if operation == 6:
            input, mlp_linear_weight = kwargs['output'], weight["linear_mlp_weight"]
            output, activation_dict["linear_mlp_input"] = dcfunc.gemm_mlp_linear(config,(input, mlp_linear_weight),stream)
        if operation == 7:
            qkv, cu_seqlens_q, cu_seqlens_kv = kwargs['output'],weight["cu_seqlens_q"],weight["cu_seqlens_kv"]
            output\
            ,activation_dict["q_padded"]\
            ,activation_dict["k_padded"]\
            ,activation_dict["v_padded"]\
            ,activation_dict["out_padded"]\
            ,activation_dict["rng_state"]\
            ,activation_dict["softmax_lse"] = dcfunc.attn(config,(qkv, cu_seqlens_q, cu_seqlens_kv),stream)
        
        if operation == 29:
            input, cu_seqlens_q, cu_seqlens_kv = kwargs['output'],weight["cu_seqlens_q"],weight["cu_seqlens_kv"]
            output = dcfunc.ring_attn_fwd(config,(input,cu_seqlens_q,cu_seqlens_kv),stream)
            if input[-1]== config.context_parallel_size-1:
                output,\
                activation_dict["q"],\
                activation_dict["kv"],\
                activation_dict["softmax_lse"],\
                activation_dict["rng_states"] = output
                
                activation_dict["out"] = output
                '''
                _\
                ,activation_dict["q_padded"]\
                ,activation_dict["k_padded"]\
                ,activation_dict["v_padded"]\
                ,activation_dict["out_padded"]\
                ,activation_dict["rng_state"]\
                ,activation_dict["softmax_lse"] = dcfunc.attn(config,(torch.randn(config.micro_batch_size*config.seq_length,3*config.hidden_size//config.tp_size,device=torch.cuda.current_device(),dtype=torch.float16), cu_seqlens_q, cu_seqlens_kv),stream) 
                '''    
                
        if operation == 30:
            # q,kv,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_kv= activation_dict["q"], activation_dict["kv"],activation_dict["softmax_lse"], activation_dict["rng_states"], weight["cu_seqlens_q"], weight["cu_seqlens_kv"]
            input,out,q,kv,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_kv= kwargs['output'],activation_dict["out"],activation_dict["q"], activation_dict["kv"],activation_dict["softmax_lse"], activation_dict["rng_states"], weight["cu_seqlens_q"], weight["cu_seqlens_kv"]

            output = dcfunc.ring_attn_bwd(config,(input,out,q,kv,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_kv),stream)

            if input[-1] == config.context_parallel_size:
                activation_dict.pop("out",None)
                activation_dict.pop("q",None)
                activation_dict.pop("kv",None)  
                activation_dict.pop("softmax_lse",None)  
                activation_dict.pop("rng_states",None)  
                activation_dict["attn_dgrad"] = output

        if operation == 8:
            input, residual = kwargs['output'], activation_dict["ln_mlp_residual"]
            output,activation_dict["mlp_mask"] = dcfunc.bda_mlp(config,(input, residual),stream)
            activation_dict.pop('ln_mlp_residual',None)
        if operation == 9:
            input, residual = kwargs['output'], activation_dict["ln_attn_residual"]
            output, activation_dict["attn_mask"] = dcfunc.bda_attn(config,(input, residual),stream)
            activation_dict.pop('ln_attn_residual',None)
        if operation == 10:
            input = kwargs['output']
            output = dcfunc.reduce_scatter(config,(input,True),stream)
        if operation == 11:
            input = kwargs['output']
            output = dcfunc.allgather(config,(input,True),stream)
        if operation == 12:
            input = kwargs['output']
            output = dcfunc.swiglu(config,(input,),stream)
            activation_dict["swiglu_input"] = input
        if operation == 13:
            grad_output, mlp_column_weight = kwargs['output'], weight["column_mlp_weight"]
            #assert check_available(grad_output), "13 error"
            output = dcfunc.gemm_dgrad_mlp_c(config,(grad_output, mlp_column_weight),stream)
        if operation == 17:
            grad_output,attn_column_weight = kwargs['output'],weight["column_attn_weight"]
            #assert check_available(grad_output), "17 error"
            output = dcfunc.gemm_dgrad_attn_c(config,(grad_output,attn_column_weight),stream)
        if operation == 15:
            grad_output, mlp_linear_weight = kwargs['output'], weight["linear_mlp_weight"]
            #assert check_available(grad_output), "15 error"
            output = dcfunc.gemm_dgrad_mlp_l(config,(grad_output, mlp_linear_weight),stream)
        if operation == 19:
            grad_output,attn_linear_weight = kwargs['output'],weight["linear_attn_weight"]
            #assert check_available(grad_output), "19 error"
            output = dcfunc.gemm_dgrad_attn_l(config,(grad_output,attn_linear_weight),stream)
        if operation == 14:
            input, grad_output, weight_ = activation_dict["gemm_mlp_c_input"], activation_dict["swiglu_dgrad"], weight["column_mlp_weight"]
            #assert check_available(grad_output), "14 error"
            if 'gemm_wgrad_mlp_c_n' in MOVABLE_OPERATOR:
                output = None
                GLOBAL_VARIABLES['gemm_wgrad_mlp_c_n']=([config,(input, grad_output, weight_.main_grad),stream],dcfunc.gemm_wgrad_mlp_c)
            else:
                output = dcfunc.gemm_wgrad_mlp_c(config,(input, grad_output, weight_.main_grad ),stream)
            
            activation_dict.pop("swiglu_dgrad",None)
            activation_dict.pop("gemm_mlp_c_input",None)
        if operation == 114:
            if 'gemm_wgrad_mlp_c_n' in GLOBAL_VARIABLES.keys():
                inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_mlp_c_n',None)
                func(*inputs)
            output = None
        if operation == 16:
            grad_output, input, weight_ = kwargs['output'], activation_dict["linear_mlp_input"], weight["linear_mlp_weight"]
            #assert check_available(grad_output), "16 error"
            if 'gemm_wgrad_mlp_l_n' in MOVABLE_OPERATOR:
                output = None
                GLOBAL_VARIABLES['gemm_wgrad_mlp_l_n']=([config,(grad_output, input, weight_.main_grad),stream],dcfunc.gemm_wgrad_mlp_l)
            else:
                output = dcfunc.gemm_wgrad_mlp_l(config,(grad_output, input, weight_.main_grad),stream)
            activation_dict.pop("linear_mlp_input",None)
        if operation == 116:
            if 'gemm_wgrad_mlp_l_n' in GLOBAL_VARIABLES.keys():
                inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_mlp_l_n',None)
                func(*inputs)
            output = None
        
        if operation == 18:
            input, grad_output, weight_ = activation_dict["gemm_attn_c_input"], kwargs['output'], weight["column_attn_weight"]
            #assert check_available(grad_output), "18 error"
            if 'gemm_wgrad_attn_c_n' in MOVABLE_OPERATOR:
                output=None
                GLOBAL_VARIABLES['gemm_wgrad_attn_c_n']=([config,(input, grad_output, weight_.main_grad),stream],dcfunc.gemm_wgrad_attn_c)
            else:
                output = dcfunc.gemm_wgrad_attn_c(config,(input, grad_output, weight_.main_grad),stream)
            activation_dict.pop("gemm_attn_c_input",None)
        if operation == 118:
            if 'gemm_wgrad_attn_c_n' in GLOBAL_VARIABLES.keys():
                inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_attn_c_n',None)
                func(*inputs)
            output = None
        if operation == 20:
            grad_output, input, weight_ = kwargs['output'], activation_dict["linear_attn_input"], weight["linear_attn_weight"]
            #assert check_available(grad_output), "20 error"
            if 'gemm_wgrad_attn_l_n' in MOVABLE_OPERATOR:
                output=None
                GLOBAL_VARIABLES['gemm_wgrad_attn_l_n']=([config,(grad_output, input, weight_.main_grad),stream],dcfunc.gemm_wgrad_attn_l)
            else:
                output = dcfunc.gemm_wgrad_attn_l(config,(grad_output, input, weight_.main_grad),stream)
            activation_dict.pop("linear_attn_input",None)
        if operation == 120:
            if 'gemm_wgrad_attn_l_n' in GLOBAL_VARIABLES.keys():
                inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_attn_l_n',None)
                func(*inputs)
            output = None
        if operation == 21:
            swiglu_input, grad_output = activation_dict["swiglu_input"], kwargs['output']
            '''
            assert check_available(grad_output), "21 error"
            if torch.distributed.get_rank() ==0:
                print("[jqruan]:",grad_output)
            '''
            output = dcfunc.swiglu_bwd(config,(swiglu_input, grad_output),stream)
            '''
            if torch.distributed.get_rank() ==0:
                print("[jqruan-output]:",output)
            assert check_available(output), "21 output error"
            '''
            activation_dict["swiglu_dgrad"] = output
            activation_dict.pop("swiglu_input",None)
        if operation == 22:
            grad_output, mask = kwargs['output'], activation_dict["mlp_mask"]
            output, activation_dict["mlp_residual_bwd"] = dcfunc.bda_mlp_bwd(config,(grad_output, mask),stream)
            activation_dict.pop("mlp_mask",None)
        if operation == 23:
            grad_output, mask = kwargs['output'], activation_dict["attn_mask"]
            output, activation_dict["attn_residual_bwd"] = dcfunc.bda_attn_bwd(config,(grad_output, mask),stream)
            activation_dict.pop("attn_mask",None)
        if operation == 24:
            grad_output,q_padded,k_padded,v_padded,out_padded,softmax_lse,cu_seqlens_q,cu_seqlens_kv,rng_state = kwargs['output'], activation_dict["q_padded"], activation_dict["k_padded"], activation_dict["v_padded"], activation_dict["out_padded"], activation_dict["softmax_lse"], weight["cu_seqlens_q"], weight["cu_seqlens_kv"],activation_dict["rng_state"]
            output = dcfunc.attn_bwd(config,(grad_output,q_padded,k_padded,v_padded,out_padded,softmax_lse,cu_seqlens_q,cu_seqlens_kv,rng_state),stream)
            activation_dict.pop("q_padded",None)
            activation_dict.pop("k_padded",None)  
            activation_dict.pop("v_padded",None)  
            activation_dict.pop("out_padded",None)  
            activation_dict.pop("softmax_lse",None)
            activation_dict.pop("rng_state",None)
        if operation == 25:
            output, ln_input, ln_mu, ln_rsigma, ln_weight, ln_bias, residual = kwargs['output'], activation_dict["ln_mlp_input"], activation_dict["mlp_mu"], activation_dict["mlp_rsigma"], weight["ln_mlp_weight"], weight["ln_mlp_bias"], activation_dict["mlp_residual_bwd"]
            ln_weight_main_grad, ln_bias_main_grad = ln_weight.main_grad, ln_bias.main_grad
            output = dcfunc.ln_mlp_bwd(config,(output, ln_input, ln_mu, ln_rsigma, ln_weight, ln_bias, residual, ln_weight_main_grad, ln_bias_main_grad), stream)
            activation_dict.pop("ln_mlp_input",None)
            activation_dict.pop("mlp_mu",None)
            activation_dict.pop("mlp_rsigma",None)
            activation_dict.pop("mlp_residual_bwd",None)

        if operation == 26:
            output, ln_input, ln_mu, ln_rsigma, ln_weight, ln_bias, residual = kwargs['output'], activation_dict["ln_attn_input"], activation_dict["attn_mu"], activation_dict["attn_rsigma"], weight["ln_attn_weight"], weight["ln_attn_bias"], activation_dict["attn_residual_bwd"]
            ln_weight_main_grad = ln_weight.main_grad
            ln_bias_main_grad = ln_bias.main_grad
            output = dcfunc.ln_attn_bwd(config,(output, ln_input, ln_mu, ln_rsigma, ln_weight, ln_bias, residual, ln_weight_main_grad, ln_bias_main_grad),stream)
            activation_dict.pop("ln_attn_input",None)
            activation_dict.pop("attn_mu",None)
            activation_dict.pop("attn_rsigma",None)
            activation_dict.pop("attn_residual_bwd",None)
        if operation ==27:
            input = activation_dict["ln_mlp_output"]
            output = dcfunc.allgather_ln_mlp(config,(input,True),stream)
            activation_dict.pop("ln_mlp_output",None)
        if operation ==28:
            input = activation_dict["ln_attn_output"]
            output = dcfunc.allgather_ln_attn(config,(input,True),stream)
            activation_dict.pop("ln_attn_output",None)
        
        # if operation == 101:

        return output
        
class DCOP:
    def __init__(self,pre_op,stream,num,is_first_mb,is_comm=0):
        self.pre = pre_op
        self.stream = stream
        self.op_num = num
        self.is_first_mb = is_first_mb
        self.is_comm = is_comm
        self.next_op = []
        self.handle = None
        self.output = None
    def __str__(self) -> str:
        for member in FuncChoice:
            if self.op_num == member.value:
                return member.name
    def set_next_op(self,next_op):
        self.next_op.append(next_op)
    def set_handle(self,handle):
        self.handle = handle
    def get_handle(self):
        return self.handle
    def set_output(self,output):
        self.output = output
    def get_op_num(self):
        return self.op_num
    def get_output(self):
        return self.output
    def get_next_op(self):
        return self.next_op

class DCBlock:
    def __init__(self,stream1_op:List[DCOP],stream2_op:List[DCOP],stream3_op:DCOP):
        self.stream1_op = stream1_op
        self.stream2_op = stream2_op
        self.stream3_op = stream3_op

#TODO isfirstmb
def init_DCBlocks(config = None,stream1=None,stream2=None):
    #fwd
    ln_attn = DCOP(None,stream1,1,True)
    #ln_attn -> allgather_attn
    allgather_attn = DCOP(ln_attn,None,11,False,1)
    ln_attn.set_next_op(allgather_attn)
    #allgather_attn ->gemm_attn_c
    gemm_attn_c = DCOP(allgather_attn,stream1,3,False)
    allgather_attn.set_next_op(gemm_attn_c)
    #gemm_attn_c ->fl_attn

    megatron_args = get_args()

    cp_attn_block_fwd_1 = None
    cp_attn_block_fwd_2 = None
    fl_attn = None

    if megatron_args.context_parallel_size > 1:
        #* ad-hoc code for Cp = 2
        cp_attn_block_fwd_1 = DCOP(gemm_attn_c,stream1, 29, True,1)
        gemm_attn_c.set_next_op(cp_attn_block_fwd_1)
        cp_attn_block_fwd_2 = DCOP(cp_attn_block_fwd_1,stream1,29,True)
        cp_attn_block_fwd_1.set_next_op(cp_attn_block_fwd_2)
        gemm_attn_l = DCOP(cp_attn_block_fwd_2,stream1,4,False)
        cp_attn_block_fwd_2.set_next_op(gemm_attn_l)

    else: 
        fl_attn = DCOP(gemm_attn_c,stream1,7,False)
        gemm_attn_c.set_next_op(fl_attn)
         #fl_attn->gemm_attn_l
        gemm_attn_l = DCOP(fl_attn,stream1,4,False)
        fl_attn.set_next_op(gemm_attn_l)

    #fl_attn->reduce_scatter_attn
    reduce_scatter_attn = DCOP(gemm_attn_l,None,10,False,1)
    gemm_attn_l.set_next_op(reduce_scatter_attn)
    #reduce_scatter_attn ->bda_attn
    bda_attn = DCOP(reduce_scatter_attn,stream1,9,False)
    reduce_scatter_attn.set_next_op(bda_attn)
    #bda_attn ->ln_mlp
    ln_mlp = DCOP(bda_attn,stream1,2,False)
    bda_attn.set_next_op(ln_mlp)
    #ln_mlp -> allgather_mlp
    allgather_mlp = DCOP(ln_mlp,None,11,False,1)
    ln_mlp.set_next_op(allgather_mlp)
    #allgather_mlp ->gemm_mlp_c
    gemm_mlp_c = DCOP(allgather_mlp,stream1,5,False)
    allgather_mlp.set_next_op(gemm_mlp_c)
    #gemm_mlp_c ->swiglu_mlp
    swiglu_mlp = DCOP(gemm_mlp_c,stream1,12,False)
    gemm_mlp_c.set_next_op(swiglu_mlp)
    #swiglu_mlp->gemm_mlp_l
    gemm_mlp_l = DCOP(swiglu_mlp,stream1,6,False)
    swiglu_mlp.set_next_op(gemm_mlp_l)
    #fl_mlp->reduce_scatter_mlp
    reduce_scatter_mlp = DCOP(gemm_mlp_l,None,10,False,1)
    gemm_mlp_l.set_next_op(reduce_scatter_mlp)
    #reduce_scatter_mlp ->bda_mlp
    bda_mlp = DCOP(reduce_scatter_mlp,stream1,8,False)
    reduce_scatter_mlp.set_next_op(bda_mlp)
    
    ##bwd
    bda_mlp_bwd = DCOP(None,stream2,22,False)
    allgather_mlp_bwd = DCOP(bda_mlp_bwd,None,11,False,2)
    # bda_mlp_bwd ->allgather_mlp_bwd
    bda_mlp_bwd.set_next_op(allgather_mlp_bwd)
    # allgather_mlp_bwd -> gemm_dgrad_mlp_linear
    gemm_dgrad_mlp_l = DCOP(allgather_mlp_bwd,stream2,15,False)
    allgather_mlp_bwd.set_next_op(gemm_dgrad_mlp_l)
    # gemm_dgrad_mlp_linear ->gemm_wgrad_mlp_l
    gemm_wgrad_mlp_l = DCOP(allgather_mlp_bwd,stream2,16,False)
    allgather_mlp_bwd.set_next_op(gemm_wgrad_mlp_l)
    #gemm_wgrad_mlp_l -> swiglu_bwd
    swiglu_bwd = DCOP(gemm_dgrad_mlp_l,stream2,21,False)
    gemm_dgrad_mlp_l.set_next_op(swiglu_bwd)
    # swiglu_bwd -> allgather_mlp_c_bwd
    # allgather_mlp_c_bwd = DCOP(swiglu_bwd,None,27,False,True)
    # swiglu_bwd.set_next_op(allgather_mlp_c_bwd)
    # swiglu_bwd -> gemm_dgrad_mlp_c
    gemm_dgrad_mlp_c = DCOP(swiglu_bwd,stream2,13,False)
    swiglu_bwd.set_next_op(gemm_dgrad_mlp_c)
    # allgather_mlp_c_bwd -> gemm_wgrad_mlp_c
    gemm_wgrad_mlp_c = DCOP(swiglu_bwd,stream2,14,False)
    swiglu_bwd.set_next_op(gemm_wgrad_mlp_c)
    # gemm_dgrad_mlp_c ->reduce_scatter_mlp_c_bwd
    reduce_scatter_mlp_c_bwd = DCOP(gemm_dgrad_mlp_c,None,10,False,2)
    gemm_dgrad_mlp_c.set_next_op(reduce_scatter_mlp_c_bwd)
    ## reduce_scatter_mlp_c_bwd -> ln_mlp_bwd
    ln_mlp_bwd = DCOP(reduce_scatter_mlp_c_bwd,stream2,25,False)
    reduce_scatter_mlp_c_bwd.set_next_op(ln_mlp_bwd)
    ## ln_mlp_bwd->bda_attn_bwd
    bda_attn_bwd = DCOP(ln_mlp_bwd,stream2,23,False)
    ln_mlp_bwd.set_next_op(bda_attn_bwd)
    # bda_attn_bwd-> allgather_attn_bwd
    allgather_attn_bwd = DCOP(bda_attn_bwd,None,11,False,2)
    bda_attn_bwd.set_next_op(allgather_attn_bwd)
    # allgather_attn_bwd -> gemm_dgrad_attn_l
    gemm_dgrad_attn_l = DCOP(allgather_attn_bwd,stream2,19,False)
    allgather_attn_bwd.set_next_op(gemm_dgrad_attn_l)
    # gemm_dgrad_attn_l -> gemm_wgrad_attn_l
    gemm_wgrad_attn_l = DCOP(gemm_dgrad_attn_l,stream2,20,False)
    allgather_attn_bwd.set_next_op(gemm_wgrad_attn_l)

    cp_attn_block_bwd_1=None
    cp_attn_block_bwd_2=None
    cp_attn_block_bwd_3=None
    attn_bwd = None

    if megatron_args.context_parallel_size >1:
        #* ad-hoc code for Cp = 2
        cp_attn_block_bwd_1 = DCOP(gemm_dgrad_attn_l,stream2, 30,False,2)
        gemm_dgrad_attn_l.set_next_op(cp_attn_block_bwd_1)
        cp_attn_block_bwd_2 = DCOP(cp_attn_block_bwd_1,stream2,30,False,2)
        cp_attn_block_bwd_1.set_next_op(cp_attn_block_bwd_2)
        cp_attn_block_bwd_3 = DCOP(cp_attn_block_bwd_2,stream2,30,False)
        cp_attn_block_bwd_2.set_next_op(cp_attn_block_bwd_3)
        gemm_dgrad_attn_c = DCOP(cp_attn_block_bwd_3,stream2,17,False)
        cp_attn_block_bwd_3.set_next_op(gemm_dgrad_attn_c)
        # allgather_attn_c_bwd -> gemm_wgrad_attn_c
        gemm_wgrad_attn_c = DCOP(cp_attn_block_bwd_3,stream2,18,False)
        cp_attn_block_bwd_3.set_next_op(gemm_wgrad_attn_c)
    
    else:
        # gemm_wgrad_attn_l -> attn_bwd
        attn_bwd = DCOP(gemm_dgrad_attn_l,stream2,24,False)
        gemm_dgrad_attn_l.set_next_op(attn_bwd)
        # attn_bwd ->allgather_attn_c_bwd
        # allgather_attn_c_bwd = DCOP(attn_bwd,None,28,False,True)
        # attn_bwd.set_next_op(allgather_attn_c_bwd)
        # attn_bwd -> gemm_dgrad_attn_c
        gemm_dgrad_attn_c = DCOP(attn_bwd,stream2,17,False)
        attn_bwd.set_next_op(gemm_dgrad_attn_c)
        # allgather_attn_c_bwd -> gemm_wgrad_attn_c
        gemm_wgrad_attn_c = DCOP(attn_bwd,stream2,18,False)
        attn_bwd.set_next_op(gemm_wgrad_attn_c)
    # gemm_dgrad_attn_c -> reduce_scatter_attn_c_bwd
    reduce_scatter_attn_c_bwd = DCOP(gemm_dgrad_attn_c,None,10,False,2)
    gemm_dgrad_attn_c.set_next_op(reduce_scatter_attn_c_bwd)
    # reduce_scatter_attn_c_bwd -> ln_attn_bwd
    ln_attn_bwd = DCOP(reduce_scatter_attn_c_bwd,stream2,26,False)
    reduce_scatter_attn_c_bwd.set_next_op(ln_attn_bwd)
    
    placement_op1 = DCOP(None, stream1, 114, False, 0)
    placement_op2 = DCOP(None, stream1, 116, False, 0)
    placement_op3 = DCOP(None, stream1, 118, False, 0)
    placement_op4 = DCOP(None, stream1, 120, False, 0)

    string_to_OP ={
        "ln_attn":ln_attn,
        "bda_mlp_bwd":bda_mlp_bwd,
        "allgather_attn":allgather_attn,
        "gemm_attn_c":gemm_attn_c,
        "fl_attn":fl_attn,
        "allgather_mlp_bwd":allgather_mlp_bwd,
        "gemm_attn_l":gemm_attn_l,
        "gemm_dgrad_mlp_l":gemm_dgrad_mlp_l,
        "gemm_wgrad_mlp_l":gemm_wgrad_mlp_l,
        "reduce_scatter_attn":reduce_scatter_attn,
        "swiglu_bwd":swiglu_bwd,
        "gemm_dgrad_mlp_c":gemm_dgrad_mlp_c,
        # "allgather_mlp_c_bwd":allgather_mlp_c_bwd,
        "gemm_wgrad_mlp_c":gemm_wgrad_mlp_c,
        "reduce_scatter_mlp_c_bwd":reduce_scatter_mlp_c_bwd,
        "bda_attn":bda_attn,
        "ln_mlp":ln_mlp,
        "ln_mlp_bwd":ln_mlp_bwd,
        "bda_attn_bwd":bda_attn_bwd,
        "allgather_mlp":allgather_mlp,
        "gemm_mlp_c":gemm_mlp_c,
        "allgather_attn_bwd":allgather_attn_bwd,
        "swiglu_mlp":swiglu_mlp,
        "gemm_mlp_l":gemm_mlp_l,
        "gemm_dgrad_attn_l":gemm_dgrad_attn_l,
        "gemm_wgrad_attn_l":gemm_wgrad_attn_l,
        "attn_bwd":attn_bwd,
        "reduce_scatter_mlp":reduce_scatter_mlp,
        "bda_mlp":bda_mlp,
        "gemm_dgrad_attn_c":gemm_dgrad_attn_c,
        # "allgather_attn_c_bwd":allgather_attn_c_bwd,
        "gemm_wgrad_attn_c":gemm_wgrad_attn_c,
        "reduce_scatter_attn_c_bwd":reduce_scatter_attn_c_bwd,
        "ln_attn_bwd":ln_attn_bwd,
        "None":None ,
        "cp_attn_block_fwd_1" : cp_attn_block_fwd_1,
        "cp_attn_block_fwd_2" : cp_attn_block_fwd_2,    #TODO ad-hoc way
        "cp_attn_block_bwd_1" : cp_attn_block_bwd_1,
        "cp_attn_block_bwd_2" : cp_attn_block_bwd_2,
        "cp_attn_block_bwd_3" : cp_attn_block_bwd_3,
        "gemm_wgrad_mlp_c_n" : placement_op1,
        "gemm_wgrad_mlp_l_n" : placement_op2,
        "gemm_wgrad_attn_c_n" : placement_op3,
        "gemm_wgrad_attn_l_n" : placement_op4,
    }
    dcblocks = None
    fwd_only = None
    bwd_only = None
    # print_rank0(f"config read {config}")
    if config is not None:
        dcblocks = []
        fwd_only = []
        bwd_only = []
        fwd_only_json = None
        bwd_only_json = None
        with open(config, 'r') as file:
            data = json.load(file)
        dc_blocks_json = data['DCBlocks']
        if 'fwd_only' in data:
            fwd_only_json = data['fwd_only']
        if 'bwd_only' in data:
            bwd_only_json = data['bwd_only']
        if dc_blocks_json is not None:
            # print_rank0(f"config read")
            for i,dc_blocks_array in enumerate(dc_blocks_json):
                stream1_ops_array = dc_blocks_array[0]
                stream2_ops_array = dc_blocks_array[1]
                stream3_ops_array = dc_blocks_array[2]
                stream1_ops = []
                stream2_ops = []
                stream3_ops = []
                # print_rank0(stream2_ops_array)
                if isinstance(stream1_ops_array,list):
                    for stream1_op in stream1_ops_array:
                        stream1_ops.append(string_to_OP[stream1_op])
                elif isinstance(stream1_ops_array,str):
                    if stream1_ops_array =="None":
                        stream1_ops = None
                    else:
                        stream1_ops.append(string_to_OP[stream1_ops_array])
                else:
                    stream1_ops = None
                if isinstance(stream2_ops_array,list):
                    for stream2_op in stream2_ops_array:
                        # print_rank0(stream2_op)
                        if stream2_op in ["gemm_wgrad_mlp_c_n","gemm_wgrad_mlp_l_n","gemm_wgrad_attn_c_n","gemm_wgrad_attn_l_n"]:
                            MOVABLE_OPERATOR.append(stream2_op)
                        stream2_ops.append(string_to_OP[stream2_op])
                elif isinstance(stream2_ops_array,str):
                    if stream2_ops_array =="None":
                        stream2_ops = None
                    else:
                        if stream2_ops_array in ["gemm_wgrad_mlp_c_n","gemm_wgrad_mlp_l_n","gemm_wgrad_attn_c_n","gemm_wgrad_attn_l_n"]:
                            MOVABLE_OPERATOR.append(stream2_ops_array)
                        stream2_ops.append(string_to_OP[stream2_ops_array])
                else:
                    stream2_ops = None
                if isinstance(stream3_ops_array,list):
                    for stream3_op in stream3_ops_array:
                        # print_rank0(stream2_op)
                        stream3_ops.append(string_to_OP[stream3_op])
                elif isinstance(stream3_ops_array,str):
                    if stream3_ops_array =="None":
                        stream3_ops = None
                    else:
                        stream3_ops.append(string_to_OP[stream3_ops_array])
                else:
                    stream3_ops = None
                dcblock = DCBlock(stream1_ops,stream2_ops,stream3_ops)
                dcblocks.append(dcblock)
                # print_rank0(f'stream1_ops: {[str(each) for each in stream1_ops] if stream1_ops is not None else "None"} stream2_ops: {[str(each) for each in stream2_ops] if stream2_ops is not None else "None"} stream3_ops: {[str(each) for each in stream3_ops] if stream3_ops is not None else "None"}')
            # exit()
        if fwd_only_json is not None:
            for i,fwd_only_array in enumerate(fwd_only_json):
                stream1_ops_array = fwd_only_array[0]
                stream2_ops_array = fwd_only_array[1]
                stream3_ops_array = fwd_only_array[2]
                stream1_ops = []
                stream2_ops = []
                stream3_ops = []
                if isinstance(stream1_ops_array,list):
                    for stream1_op in stream1_ops_array:
                        stream1_ops.append(string_to_OP[stream1_op])
                elif isinstance(stream1_ops_array,str):
                    if stream1_ops_array =="None":
                        stream1_ops = None
                    else:
                        stream1_ops.append(string_to_OP[stream1_ops_array])
                else:
                    stream1_ops = None
                if isinstance(stream2_ops_array,list):
                    for stream2_op in stream2_ops_array:
                        # print_rank0(stream2_op)
                        stream2_ops.append(string_to_OP[stream2_op])
                elif isinstance(stream2_ops_array,str):
                    if stream2_ops_array =="None":
                        stream2_ops = None
                    else:
                        stream2_ops.append(string_to_OP[stream2_ops_array])
                else:
                    stream2_ops = None
                if isinstance(stream3_ops_array,list):
                    for stream3_op in stream3_ops_array:
                        # print_rank0(stream2_op)
                        stream3_ops.append(string_to_OP[stream3_op])
                elif isinstance(stream3_ops_array,str):
                    if stream3_ops_array =="None":
                        stream3_ops = None
                    else:
                        stream3_ops.append(string_to_OP[stream3_ops_array])
                else:
                    stream3_ops = None
                dcblock = DCBlock(stream1_ops,stream2_ops,stream3_ops)
                fwd_only.append(dcblock)
        if bwd_only_json is not None:
            for i,bwd_only_array in enumerate(bwd_only_json):
                stream1_ops_array = bwd_only_array[0]
                stream2_ops_array = bwd_only_array[1]
                stream3_ops_array = bwd_only_array[2]
                stream1_ops = []
                stream2_ops = []
                stream3_ops = []
                if isinstance(stream1_ops_array,list):
                    for stream1_op in stream1_ops_array:
                        stream1_ops.append(string_to_OP[stream1_op])
                elif isinstance(stream1_ops_array,str):
                    if stream1_ops_array =="None":
                        stream1_ops = None
                    else:
                        stream1_ops.append(string_to_OP[stream1_ops_array])
                else:
                    stream1_ops = None
                if isinstance(stream2_ops_array,list):
                    for stream2_op in stream2_ops_array:
                        # print_rank0(stream2_op)
                        stream2_ops.append(string_to_OP[stream2_op])
                elif isinstance(stream2_ops_array,str):
                    if stream2_ops_array =="None":
                        stream2_ops = None
                    else:
                        stream2_ops.append(string_to_OP[stream2_ops_array])
                else:
                    stream2_ops = None
                if isinstance(stream3_ops_array,list):
                    for stream3_op in stream3_ops_array:
                        # print_rank0(stream2_op)
                        stream3_ops.append(string_to_OP[stream3_op])
                elif isinstance(stream3_ops_array,str):
                    if stream3_ops_array =="None":
                        stream3_ops = None
                    else:
                        stream3_ops.append(string_to_OP[stream3_ops_array])
                else:
                    stream3_ops = None
                dcblock = DCBlock(stream1_ops,stream2_ops,stream3_ops)
                bwd_only.append(dcblock)
    if len(dcblocks) ==0:
        dcblocks = [DCBlock([ln_attn],None,None),
                    DCBlock(None,[bda_mlp_bwd],[allgather_attn]),
                    DCBlock([gemm_attn_c,fl_attn],None,[allgather_mlp_bwd]),
                    DCBlock([gemm_attn_l],None,None),
                    DCBlock(None,[gemm_dgrad_mlp_l,gemm_wgrad_mlp_l],[reduce_scatter_attn]),
                    DCBlock(None,[swiglu_bwd],None),
                    DCBlock(None,[gemm_dgrad_mlp_c],None),
                    DCBlock(None,[gemm_wgrad_mlp_c],[reduce_scatter_mlp_c_bwd]),
                    DCBlock([bda_attn,ln_mlp],None,None),
                    DCBlock(None,[ln_mlp_bwd,bda_attn_bwd],[allgather_mlp]),
                    DCBlock([gemm_mlp_c],None,[allgather_attn_bwd]),
                    DCBlock([swiglu_mlp,gemm_mlp_l],None,None),
                    DCBlock(None,[gemm_dgrad_attn_l,gemm_wgrad_attn_l,attn_bwd],[reduce_scatter_mlp]),
                    DCBlock([bda_mlp],None,None),
                    DCBlock(None,[gemm_dgrad_attn_c],None),
                    DCBlock(None,[gemm_wgrad_attn_c],[reduce_scatter_attn_c_bwd]),
                    DCBlock(None,[ln_attn_bwd],None)
                    ]
    if len(fwd_only) == 0 :
        if megatron_args.context_parallel_size > 1:
            fwd_only = [
                #fwd
                DCBlock([ln_attn],None,None),
                DCBlock(None,None,[allgather_attn]),
                DCBlock([gemm_attn_c],None,None),
                DCBlock(None,None,[cp_attn_block_fwd_1]),
                DCBlock([cp_attn_block_fwd_2],None,None),
                DCBlock([gemm_attn_l],None,None),
                DCBlock(None,None,[reduce_scatter_attn]),
                DCBlock([bda_attn,ln_mlp],None,None),
                DCBlock(None,None,[allgather_mlp]),
                DCBlock([gemm_mlp_c],None,None),
                DCBlock([swiglu_mlp,gemm_mlp_l],None,None),
                DCBlock(None,None,[reduce_scatter_mlp]),
                DCBlock([bda_mlp],None,None),
            ]
        else:
            fwd_only = [
                #fwd
                DCBlock([ln_attn],None,None),
                DCBlock(None,None,[allgather_attn]),
                DCBlock([gemm_attn_c,fl_attn],None,None),
                DCBlock([gemm_attn_l],None,None),
                DCBlock(None,None,[reduce_scatter_attn]),
                DCBlock([bda_attn,ln_mlp],None,None),
                DCBlock(None,None,[allgather_mlp]),
                DCBlock([gemm_mlp_c],None,None),
                DCBlock([swiglu_mlp,gemm_mlp_l],None,None),
                DCBlock(None,None,[reduce_scatter_mlp]),
                DCBlock([bda_mlp],None,None),
            ]
    if len(bwd_only) ==0 :
        if megatron_args.context_parallel_size > 1:
            bwd_only=[
                #bwd
                DCBlock(None,[bda_mlp_bwd],None),
                DCBlock(None,None,[allgather_mlp_bwd]),
                DCBlock(None,[gemm_dgrad_mlp_l,gemm_wgrad_mlp_l],None),
                DCBlock(None,[swiglu_bwd],None),
                DCBlock(None,[gemm_dgrad_mlp_c],None),
                DCBlock(None,[gemm_wgrad_mlp_c],[reduce_scatter_mlp_c_bwd]),
                DCBlock(None,[ln_mlp_bwd,bda_attn_bwd],None),
                DCBlock(None,None,[allgather_attn_bwd]),
                DCBlock(None,[gemm_dgrad_attn_l,gemm_wgrad_attn_l],None),
                DCBlock(None,None,[cp_attn_block_bwd_1]),
                DCBlock(None,None,[cp_attn_block_bwd_2]),
                DCBlock(None,[cp_attn_block_bwd_3],None),
                DCBlock(None,[gemm_dgrad_attn_c],None),
                DCBlock(None,[gemm_wgrad_attn_c],[reduce_scatter_attn_c_bwd]),
                DCBlock(None,[ln_attn_bwd],None)
                ]
        else:
            bwd_only=[
                #bwd
                DCBlock(None,[bda_mlp_bwd],None),
                DCBlock(None,None,[allgather_mlp_bwd]),
                DCBlock(None,[gemm_dgrad_mlp_l,gemm_wgrad_mlp_l],None),
                DCBlock(None,[swiglu_bwd],None),
                DCBlock(None,[gemm_dgrad_mlp_c],None),
                DCBlock(None,[gemm_wgrad_mlp_c],[reduce_scatter_mlp_c_bwd]),
                DCBlock(None,[ln_mlp_bwd,bda_attn_bwd],None),
                DCBlock(None,None,[allgather_attn_bwd]),
                DCBlock(None,[gemm_dgrad_attn_l,gemm_wgrad_attn_l,attn_bwd],None),
                DCBlock(None,[gemm_dgrad_attn_c],None),
                DCBlock(None,[gemm_wgrad_attn_c],[reduce_scatter_attn_c_bwd]),
                DCBlock(None,[ln_attn_bwd],None)
                ]

    return dcblocks,fwd_only,bwd_only

def exec(config,dcoperation,fwd_weight,bwd_weight,input,grad_output,dcblocks:List[DCBlock],stream1,stream2,activation_dict,new_activation_dict):
    fwd_result = None
    bwd_result = None
    for i,now_block in enumerate(dcblocks):
        fwd_wait = False
        bwd_wait = False
        handle_status = False
        # stream3
        # print_rank0(f'stream1_ops: {[str(each) for each in now_block.stream1_op] if now_block.stream1_op is not None else "None"} stream2_ops: {[str(each) for each in now_block.stream2_op] if now_block.stream2_op is not None else "None"} stream3_ops: {[str(each) for each in now_block.stream3_op] if now_block.stream3_op is not None else "None"}')
        if now_block.stream3_op!=None:
            now_ops = now_block.stream3_op
            # if config.tensor_model_parallel_size == 1:
            #     handle_status = False
            #     output = now_op.get_output()
            # else:
            #TODO CP op needs stream
            '''
            if dist.get_rank() ==0:
                print("[jqruan] op num: ", now_op.get_op_num())
            '''
            #TODO consider adding fwd/bwd info in comm ops
            for _,now_op in enumerate(now_ops):
                if now_op.is_comm == 1:
                    fwd_wait = True
                elif now_op.is_comm == 2:
                    bwd_wait = True

                if fwd_weight is None:
                    output,handle = dcoperation.perform_operation(DCFunc,now_op.get_op_num(),config,bwd_weight,activation_dict,output=now_op.get_output(),stream=now_op.stream)
                else:
                    output,handle = dcoperation.perform_operation(DCFunc,now_op.get_op_num(),config,fwd_weight,activation_dict,output=now_op.get_output(),stream=now_op.stream)
                
                now_op.set_output(None)
                if config.tensor_model_parallel_size >1 or config.context_parallel_size >1:
                    handle_status = True
                # if handle is not None:
                #     handle.wait()
                next_op = now_op.get_next_op()
                for nop in next_op:
                    nop.set_output(output)
                    nop.set_handle(handle)  
        if fwd_wait :
            #stream2 bwd
            if now_block.stream2_op!=None:
                now_ops = now_block.stream2_op
                for _,now_op in enumerate(now_ops):
                    if now_op.get_handle()!=None:
                        
                        if isinstance(now_op.get_handle(),list):
                            for comm_req in now_op.get_handle():
                                comm_req.wait()
                        else:
                            now_op.get_handle().wait()

                        # torch.cuda.synchronize()
                        now_op.set_handle(None)
                    out = grad_output if now_op.get_output() is None else now_op.get_output()
                    output = dcoperation.perform_operation(DCFunc,now_op.get_op_num(),config,bwd_weight,activation_dict,output=out,stream=now_op.stream)
                    now_op.set_output(None)
                    next_op = now_op.get_next_op()
                    if output is not None and len(next_op)==0:
                        bwd_result = output
                    for nop in next_op:
                        nop.set_output(output)
            #stream1 fwd
            if now_block.stream1_op!=None:
                now_ops = now_block.stream1_op
                for _,now_op in enumerate(now_ops):
                    if now_op.get_handle()!=None:
                        
                        if isinstance(now_op.get_handle(),list):
                            for comm_req in now_op.get_handle():
                                comm_req.wait()
                        else:
                            now_op.get_handle().wait()

                        #torch.cuda.synchronize()
                        now_op.set_handle(None)
                    out = input if now_op.get_output() is None else now_op.get_output()
                    output = dcoperation.perform_operation(DCFunc,now_op.get_op_num(),config,fwd_weight,new_activation_dict,output=out,stream=now_op.stream)
                    now_op.set_output(None)
                    next_op = now_op.get_next_op()
                    if output is not None and len(next_op)==0:
                        fwd_result = output
                    for nop in next_op:
                        nop.set_output(output)
        else:
            #stream1 fwd
            if now_block.stream1_op!=None:
                now_ops = now_block.stream1_op
                for _,now_op in enumerate(now_ops):
                    if now_op.get_handle()!=None:
                        
                        if isinstance(now_op.get_handle(),list):
                            for comm_req in now_op.get_handle():
                                comm_req.wait()
                        else:
                            now_op.get_handle().wait()

                        #torch.cuda.synchronize()
                        now_op.set_handle(None)
                    out = input if now_op.get_output() is None else now_op.get_output()
                    output = dcoperation.perform_operation(DCFunc,now_op.get_op_num(),config,fwd_weight,new_activation_dict,output=out,stream=now_op.stream)
                    now_op.set_output(None)
                    next_op = now_op.get_next_op()
                    if output is not None and len(next_op)==0:
                        fwd_result = output
                    for nop in next_op:
                        nop.set_output(output)
            
            #stream2 bwd
            if now_block.stream2_op!=None:
                now_ops = now_block.stream2_op
                for _,now_op in enumerate(now_ops):
                    if now_op.get_handle()!=None:
                        
                        if isinstance(now_op.get_handle(),list):
                            for comm_req in now_op.get_handle():
                                comm_req.wait()
                        else:
                            now_op.get_handle().wait()

                        # torch.cuda.synchronize()
                        now_op.set_handle(None)
                    out = grad_output if now_op.get_output() is None else now_op.get_output()
                    output = dcoperation.perform_operation(DCFunc,now_op.get_op_num(),config,bwd_weight,activation_dict,output=out,stream=now_op.stream)
                    now_op.set_output(None)
                    next_op = now_op.get_next_op()
                    if output is not None and len(next_op)==0:
                        bwd_result = output
                    for nop in next_op:
                        nop.set_output(output)
        # if handle_status:
        torch.cuda.synchronize()

        if 'gemm_wgrad_mlp_c_n' in GLOBAL_VARIABLES.keys() and len(new_activation_dict)==0:
            inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_mlp_c_n',None)
            func(*inputs)
        if 'gemm_wgrad_mlp_l_n' in GLOBAL_VARIABLES.keys() and len(new_activation_dict)==0:
            inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_mlp_l_n',None)
            func(*inputs)
        if 'gemm_wgrad_attn_c_n' in GLOBAL_VARIABLES.keys() and len(new_activation_dict)==0:
            inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_attn_c_n',None)
            func(*inputs)
        if 'gemm_wgrad_attn_l_n' in GLOBAL_VARIABLES.keys() and len(new_activation_dict)==0:
            inputs, func = GLOBAL_VARIABLES.pop('gemm_wgrad_attn_l_n',None)
            func(*inputs)

    return fwd_result,bwd_result

def deallocate_tensor(tensor, enable_deallocate_tensor=False):
    '''
    At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (tensor is None) or (not enable_deallocate_tensor):
        return
    assert isinstance(tensor, torch.Tensor), "expected Tensor, found %s." % type(tensor).__name__
    assert tensor._base is None, "counter-productive to free a view of another tensor."
    tensor.data = torch.empty((1,), device=tensor.device, dtype=tensor.dtype,)

class DSTB_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder,config, bwd_weight, fwd_weight,schedule,stream1,stream2):
        bwd_input, fwd_input = GLOBAL_VARIABLES['inputs']
        ctx.fwd_weight = fwd_weight
        ctx.bwd_weight = bwd_weight
        ctx.schedule = schedule
        ctx.stream1 = stream1
        ctx.stream2 = stream2
        ctx.config = config
        activation_dict_new = {}
        activation_dict = {}
        if bwd_input is None:
            with torch.no_grad():
                # only fwd this mb
                fwd_result, _ = exec(config,DCoperation,fwd_weight,None,fwd_input,None,schedule,stream1,stream2,{},activation_dict_new)
                GLOBAL_VARIABLES['fwd_activations'][-1].push(activation_dict_new)
                GLOBAL_VARIABLES['inputs']=[None,fwd_result]
            return placeholder
        # get activations of the last micro-batch firstly
        activation_dict = GLOBAL_VARIABLES['bwd_activations'][0].pop()
        if GLOBAL_VARIABLES['bwd_activations'][0].is_empty():
            GLOBAL_VARIABLES['bwd_activations'].pop(0)
        if fwd_input is None:
            with torch.no_grad():
                # only bwd last mb
                _ , bwd_result = exec(config,DCoperation,None,bwd_weight,None,bwd_input,schedule,stream1,stream2,activation_dict,{})
                GLOBAL_VARIABLES['inputs']=[bwd_result,None]
            return placeholder
        with torch.no_grad():
            # foward and backward pass here!
            fwd_result , bwd_result = exec(config,DCoperation,fwd_weight,bwd_weight,fwd_input,bwd_input,schedule,stream1,stream2,activation_dict,activation_dict_new)
            GLOBAL_VARIABLES['fwd_activations'][-1].push(activation_dict_new)
            GLOBAL_VARIABLES['inputs']=[bwd_result,fwd_result]
        return placeholder
    
    @staticmethod
    def backward(ctx,*args):
        bwd_input, fwd_input = GLOBAL_VARIABLES['inputs']
        # need to exchange the positions
        bwd_weight = ctx.fwd_weight
        fwd_weight = ctx.bwd_weight
        schedule = ctx.schedule
        stream1 = ctx.stream1
        stream2 = ctx.stream2
        config = ctx.config
        activation_dict_new = {}
        activation_dict = {}
        if bwd_input is None:
            with torch.no_grad():
                # only fwd this mb
                fwd_result, _ = exec(config,DCoperation,fwd_weight,None,fwd_input,None,schedule,stream1,stream2,{},activation_dict_new)
                GLOBAL_VARIABLES['bwd_activations'][-1].push(activation_dict_new)
                GLOBAL_VARIABLES['inputs']=[None,fwd_result]
            return None,None,None,None,None,None,None

        activation_dict = GLOBAL_VARIABLES['fwd_activations'][0].pop()
        if GLOBAL_VARIABLES['fwd_activations'][0].is_empty():
            GLOBAL_VARIABLES['fwd_activations'].pop(0)
        if fwd_input is None:
            with torch.no_grad():
                # only bwd
                _ , bwd_result = exec(config,DCoperation,None,bwd_weight,None,bwd_input,schedule,stream1,stream2,activation_dict,{})
                GLOBAL_VARIABLES['inputs']=[bwd_result,None]
            return None,None,None,None,None,None,None
        with torch.no_grad():
            # foward and backward pass here!
            fwd_result , bwd_result = exec(config,DCoperation,fwd_weight,bwd_weight,fwd_input,bwd_input,schedule,stream1,stream2,activation_dict,activation_dict_new)
            GLOBAL_VARIABLES['bwd_activations'][-1].push(activation_dict_new)
            GLOBAL_VARIABLES['inputs']=[bwd_result,fwd_result]
        return None,None,None,None,None,None,None
#TODO bwd weight need to be related to fwd weight
class DSTB(torch.nn.Module):
    def __init__(self, config,stream1,stream2) -> None:
        args = get_args()
        super(DSTB, self).__init__()
        #first layer means fwd weight
        self.stream1 = stream1
        self.stream2 = stream2
        self.ln_attn_weight_first = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.column_attn_weight_first = Parameter(torch.empty(int((config.hidden_size*3)/config.tensor_model_parallel_size),config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.linear_attn_weight_first = Parameter(torch.empty(config.hidden_size,int(config.hidden_size/config.tensor_model_parallel_size),dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_attn_bias_first = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_mlp_weight_first = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.column_mlp_weight_first = Parameter(torch.empty(int((2*config.ffn_hidden_size)/config.tensor_model_parallel_size),config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.linear_mlp_weight_first = Parameter(torch.empty(config.hidden_size,int(config.ffn_hidden_size/config.tensor_model_parallel_size),dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_mlp_bias_first = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        #second layer means last mb fwd weight which uses for bwd
        self.ln_attn_weight_second = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.column_attn_weight_second = Parameter(torch.empty(int((config.hidden_size*3)/config.tensor_model_parallel_size),config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.linear_attn_weight_second = Parameter(torch.empty(config.hidden_size,int(config.hidden_size/config.tensor_model_parallel_size),dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_attn_bias_second = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_mlp_weight_second = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.column_mlp_weight_second = Parameter(torch.empty(int((2*config.ffn_hidden_size)/config.tensor_model_parallel_size),config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.linear_mlp_weight_second = Parameter(torch.empty(config.hidden_size,int(config.ffn_hidden_size/config.tensor_model_parallel_size),dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_mlp_bias_second = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        # init parameters
        # config.init_method(self.ln_attn_weight_first)
        with torch.no_grad():
            self.ln_attn_weight_first.fill_(float(True))
            self.ln_mlp_weight_first.fill_(float(True))
            self.ln_attn_bias_first.zero_()
            self.ln_mlp_bias_first.zero_()

        config.init_method(self.column_attn_weight_first)
        config.init_method(self.linear_attn_weight_first)
        config.init_method(self.column_mlp_weight_first)
        config.init_method(self.linear_mlp_weight_first)

        with torch.no_grad():
            self.ln_attn_weight_second.fill_(float(True))
            self.ln_mlp_weight_second.fill_(float(True))
            self.ln_attn_bias_second.zero_()
            self.ln_mlp_bias_second.zero_()
        
        config.init_method(self.column_attn_weight_second)
        config.init_method(self.linear_attn_weight_second)
        config.init_method(self.column_mlp_weight_second)
        config.init_method(self.linear_mlp_weight_second)
        # set seq parallel flag, only layernorm layer should set this flag
        setattr(self.ln_attn_weight_first, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_mlp_weight_first, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_attn_weight_second, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_mlp_weight_second, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_attn_bias_first, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_mlp_bias_first, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_attn_bias_second, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_mlp_bias_second, 'sequence_parallel', config.sequence_parallel)
        #TODO config
        self.weight_first={}
        self.weight_first["ln_attn_weight"] = self.ln_attn_weight_first
        self.weight_first["column_attn_weight"] = self.column_attn_weight_first
        self.weight_first["linear_attn_weight"] = self.linear_attn_weight_first
        self.weight_first["ln_attn_bias"] = self.ln_attn_bias_first
        self.weight_first["ln_mlp_weight"] = self.ln_mlp_weight_first
        self.weight_first["column_mlp_weight"] = self.column_mlp_weight_first
        self.weight_first["linear_mlp_weight"] = self.linear_mlp_weight_first
        self.weight_first["ln_mlp_bias"] = self.ln_mlp_bias_first
        self.weight_first["cu_seqlens_q"] = torch.arange(0,int((args.micro_batch_size+1)*config.seq_length),step=config.seq_length, device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
        self.weight_first["cu_seqlens_kv"] = torch.arange(0,int((args.micro_batch_size+1)*config.seq_length),step=config.seq_length, device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
        self.weight_second={}
        self.weight_second["ln_attn_weight"] = self.ln_attn_weight_second
        self.weight_second["column_attn_weight"] = self.column_attn_weight_second
        self.weight_second["linear_attn_weight"] = self.linear_attn_weight_second
        self.weight_second["ln_attn_bias"] = self.ln_attn_bias_second
        self.weight_second["ln_mlp_weight"] = self.ln_mlp_weight_second
        self.weight_second["column_mlp_weight"] = self.column_mlp_weight_second
        self.weight_second["linear_mlp_weight"] = self.linear_mlp_weight_second
        self.weight_second["ln_mlp_bias"] = self.ln_mlp_bias_second
        self.weight_second["cu_seqlens_q"] = torch.arange(0,int((args.micro_batch_size+1)*config.seq_length),step=config.seq_length, device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
        self.weight_second["cu_seqlens_kv"] = torch.arange(0,int((args.micro_batch_size+1)*config.seq_length),step=config.seq_length, device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
        config.micro_batch_size = args.micro_batch_size
        # config.seq_length = args.seq_length
        self.dcblocks,self.fwd_only_dcblocks,self.bwd_only_dcblocks = init_DCBlocks(args.schedule_config,stream1,stream2)
        self.config = config

    def forward(self, inputs):
        placeholder = inputs
        # x2 means fwd_input x1 means bwd_input
        x1, x2 = GLOBAL_VARIABLES['inputs']
        if x2 is None:
            schedule = self.bwd_only_dcblocks
        elif x1 is None:
            schedule = self.fwd_only_dcblocks
        else:
            schedule = self.dcblocks
        placeholder = DSTB_.apply(placeholder,
                                  self.config, 
                                  self.weight_second, 
                                  self.weight_first,
                                  schedule,
                                  self.stream1,
                                  self.stream2)
        
        return placeholder
