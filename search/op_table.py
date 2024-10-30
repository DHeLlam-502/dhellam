OP_TABLE_IDX = {
    'allgather_attn': 0,
    'allgather_attn_bwd': 1,
    'allgather_mlp': 2,
    'allgather_mlp_bwd': 3,
    'reduce_scatter_attn': 4,
    'reduce_scatter_mlp': 5,
    'gemm_dgrad_attn_c': 6,
    'gemm_dgrad_mlp_c': 7,
    'gemm_wgrad_attn_c_reduce_scatter_attn_c_bwd': 8,
    'gemm_wgrad_mlp_c_reduce_scatter_mlp_c_bwd': 9,
    'attn_bwd': 10,
    'fl_attn': 11,
    'bda_attn': 12,
    'bda_mlp': 13,
    'bda_attn_bwd': 14,
    'bda_mlp_bwd': 15,
    'gemm_attn_c': 16,
    'gemm_attn_l': 17,
    'gemm_dgrad_attn_l': 18,
    'gemm_dgrad_mlp_l': 19,
    'gemm_mlp_c': 20,
    'gemm_mlp_l': 21,
    'gemm_wgrad_attn_l': 22,
    'gemm_wgrad_mlp_l': 23,
    'ln_attn': 24,
    'ln_mlp': 25,
    'ln_attn_bwd': 26,
    'ln_mlp_bwd': 27,
    'swiglu_bwd': 28,
    'swiglu_mlp': 29,
    'cp_attn_block_fwd_1' : 30,
    'cp_attn_block_fwd_2' : 31,
    'cp_attn_block_bwd_1' : 32,
    'cp_attn_block_bwd_2' : 33,
    'cp_attn_block_bwd_3' : 34,
}

fwd_operators_list = None
bwd_operators_list = None

def operator_list_prune(context_parallel_size=1):
    global fwd_operators_list, bwd_operators_list
    if context_parallel_size == 1:
        fwd_operators_list = [
            [24],  # ln_attn
            [0],   # allgather_attn
            [16, 11, 17],  # gemm_attn_c, fl_attn, gemm_attn_l
            [4],   # reduce_scatter_attn
            [12, 25],  # bda_attn, ln_mlp
            [2],   # allgather_mlp
            [20, 29, 21],  # gemm_mlp_c, swiglu_mlp, gemm_mlp_l
            [5],   # reduce_scatter_mlp
            [13]   # bda_mlp
        ]

        bwd_operators_list = [
            [15],  # bda_mlp_bwd
            [3],   # allgather_mlp_bwd
            [19, 23, 28, 7],  # gemm_dgrad_mlp_l, gemm_wgrad_mlp_l, swiglu_bwd, gemm_dgrad_mlp_c
            [9],   # gemm_wgrad_mlp_c_reduce_scatter_mlp_c_bwd
            [27, 14],  # ln_mlp_bwd, bda_attn_bwd
            [1],   # allgather_attn_bwd
            [18, 22, 10, 6],  # gemm_dgrad_attn_l, gemm_wgrad_attn_l, attn_bwd, gemm_dgrad_attn_c
            [8],   # gemm_wgrad_attn_c_reduce_scatter_attn_c_bwd
            [26]   # ln_attn_bwd
        ]
    elif context_parallel_size == 2:
        fwd_operators_list = [
            [24],  # ln_attn
            [0, 16], # allgather_attn, gemm_attn_c
            [30], # cp_attn_fwd1
            [31,17], # cp_attn_fwd2, gemm_attn_l
            [4,12,25], # reduce_scatter_attn, bda_attn, ln_mlp
            [2,20,29,21],  # allgather_mlp, gemm_mlp_c, swiglu_mlp, gemm_mlp_l
            [5,13],  # reduce_scatter_mlp, bda_mlp
        ]
        bwd_operators_list = [
            [15],  # bda_mlp_bwd
            [3, 19, 23, 28, 7],   # allgather_mlp_bwd, gemm_dgrad_mlp_l, gemm_wgrad_mlp_l, swiglu_bwd, gemm_dgrad_mlp_c
            [9],        # gemm_wgrad_mlp_c_reduce_scatter_mlp_c_bwd
            [27, 14],   # ln_mlp_bwd, bda_attn_bwd
            [1, 18, 22],   # allgather_attn_bwd, gemm_dgrad_attn_l, gemm_wgrad_attn_l
            [32],  # cp_attn_bwd1
            [33],  # cp_attn_bwd2
            [34,6], # cp_attn_bwd3, gemm_dgrad_attn_c
            [8],   # gemm_wgrad_attn_c_reduce_scatter_attn_c_bwd
            [26],  #ln_attn_bwd
        ]
    else:
        print(f">>>> Context Parallel Size Error: {context_parallel_size}")
    return (len(fwd_operators_list),sum([len(each) for each in fwd_operators_list]),fwd_operators_list), (len(bwd_operators_list),sum([len(each) for each in bwd_operators_list]),bwd_operators_list)

try:
    from dhellam.core.dstb import DCFunc
    import torch
    from megatron.arguments import core_transformer_config_from_args
    from megatron import get_args
    from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_context_parallel_global_ranks,
    get_context_parallel_group
    )
    args = get_args()
    B = args.micro_batch_size
    S = args.seq_length
    H = args.hidden_size
    A = args.num_attention_heads
    TP = args.tensor_model_parallel_size
    FFN = args.ffn_hidden_size
    config = core_transformer_config_from_args(args)
    config = core_transformer_config_from_args(get_args())
    config.tp_group = get_tensor_model_parallel_group(check_initialized=False)
    config.tp_size = args.tensor_model_parallel_size
    config.seq_length = S
    config.micro_batch_size = B
    config.hidden_size =H
    config.num_attention_heads =A
    config.tensor_model_parallel_size = TP
    config.ffn_hidden_size = FFN
    config.cp_group = get_context_parallel_group(check_initialized=False)
    config.cp_global_ranks = get_context_parallel_global_ranks(check_initialized=False)
except:
    print(f"warning: package not found.")

def bda_bwd_(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    mask = torch.randn(b*s//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    
    return [
        config,
        (input,mask),
        stream
    ], DCFunc.bda_mlp_bwd

def bda_(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    residual = torch.randn(b,s//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [
        config,
        (input,residual),
        stream
    ], DCFunc.bda_attn

def cp_fa_bwd(b,s,h,a,tp,ffn,stream):

    out = torch.randn(b,s,h//tp, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    q = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    kv = torch.randn(2,b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    softmax_lse = torch.randn(b, a//tp, s, dtype=torch.float32,device=f"cuda:{torch.cuda.current_device()}")
    rng_states = [ torch.tensor([1234, 96720], dtype=torch.int64 ,device=f"cuda:{torch.cuda.current_device()}") for i in range(args.context_parallel_size)]
    dout = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    dq = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    dkv_ = torch.randn(2,b*s,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    cu_seqlens_k = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)

    p2p_comm_buffers = None
    cp_iter_times = 0
    input = [dout,dq,dkv_,p2p_comm_buffers,cp_iter_times]
    return [config,(input,out,q,kv,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_k),stream], DCFunc.ring_attn_bwd

def cp_fa_bwd_last(b,s,h,a,tp,ffn,stream): 

    out = torch.randn(b,s,h//tp, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    q = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    kv = torch.randn(2,b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    softmax_lse = torch.randn(b, a//tp, s, dtype=torch.float32,device=f"cuda:{torch.cuda.current_device()}")
    rng_states = [ torch.tensor([1234, 96720], dtype=torch.int64 ,device=f"cuda:{torch.cuda.current_device()}") for i in range(args.context_parallel_size)]
    dout = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    dq = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    dkv_ = torch.randn(2,b*s,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    cu_seqlens_k = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)

    p2p_comm_buffers = [torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device), \
                        torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)]
    cp_iter_times = args.context_parallel_size
    input = [dout,dq,dkv_,p2p_comm_buffers,cp_iter_times]
    return [config,(input,out,q,kv,softmax_lse,rng_states,cu_seqlens_q,cu_seqlens_k),stream], DCFunc.ring_attn_bwd

def fa_bwd(b,s,h,a,tp,ffn,stream):
    q =    torch.randn(b*s, a//tp, h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    k =    torch.randn(b*s, a//tp, h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    v =    torch.randn(b*s, a//tp, h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    out = torch.randn(b*s, a//tp, h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    dout =  torch.randn(b*s, h//tp, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    cu_seqlens_k = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    softmax_lse = torch.randn(b, a//tp, s, dtype=torch.float32,device=f"cuda:{torch.cuda.current_device()}")
    rng_state = torch.tensor([813829799034691,           72172], device=f"cuda:{torch.cuda.current_device()}")
    return [config,(dout,q,k,v,out,softmax_lse,cu_seqlens_q,cu_seqlens_k,rng_state),stream], DCFunc.attn_bwd

def cp_fa_(b,s,h,a,tp,ffn,stream):  
    qkv = torch.randn(b*s,3*h//tp, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    cu_seqlens_k = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    return [
        config,
        ([qkv,None,None,None,None,None,None,0],cu_seqlens_q,cu_seqlens_k),
        stream
    ],DCFunc.ring_attn_fwd

def cp_fa_last(b,s,h,a,tp,ffn,stream):
    #* Causal FA
    q = torch.randn(b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    p2p_comm_buffers = [ torch.randn(2,b,2,s//2,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}") for i in range(args.context_parallel_size)]
    out_per_step = [  torch.randn(b*s,a//tp,h//a, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}") for i in range(args.context_parallel_size)]
    softmax_lse_per_step = [ torch.randn(b,a//tp,s, dtype=torch.float32,device=f"cuda:{torch.cuda.current_device()}") for i in range(args.context_parallel_size)]
    rng_states = [ torch.tensor([1234, 96720], dtype=torch.int64 ,device=f"cuda:{torch.cuda.current_device()}") for i in range(args.context_parallel_size)]
    softmax_lse = None
    softmax_lse_ = None
    cp_iter_times = args.context_parallel_size -1

    input = [q,p2p_comm_buffers,out_per_step,softmax_lse_per_step,rng_states,softmax_lse,softmax_lse_,cp_iter_times]
    cu_seqlens_k = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)

    return [
        config,
        (input,cu_seqlens_q,cu_seqlens_k),
        stream
    ],DCFunc.ring_attn_fwd

def fa_(b,s,h,a,tp,ffn,stream):
    qkv = torch.randn(b*s,3*h//tp, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    cu_seqlens_k = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    cu_seqlens_q = torch.tensor(list(range(0,b*s+1,s)), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    return [
        config,
        (qkv,cu_seqlens_q,cu_seqlens_k),
        stream
    ],DCFunc.attn

def swiglu_bwd(b,s,h,a,tp,ffn,stream):
    dgrad = torch.randn(b*s,2*h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    input = torch.randn(b*s,4*h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [
        config,
        (input,dgrad),
        stream
    ], DCFunc.swiglu_bwd

def swiglu_(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,4*h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,),stream], DCFunc.swiglu

def ln_bwd(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s//tp,h, dtype=torch.float16, device=f"cuda:{torch.cuda.current_device()}")
    doutput= torch.randn(b*s//tp,h, dtype=torch.float16, device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    ln_weight_main_grad = torch.randn(h, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    ln_bias_main_grad = torch.randn(h, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    bias =torch.randn(h, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    mu= torch.randn(b*s//tp, dtype=torch.float32,device=f"cuda:{torch.cuda.current_device()}")
    rsigma= torch.randn(b*s//tp, dtype=torch.float32,device=f"cuda:{torch.cuda.current_device()}")
    residual = torch.randn(b*s//tp,h, dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [
        config,(
        doutput,
        input,
        mu,
        rsigma,
        weight,
        bias,
        residual,
        ln_weight_main_grad,
        ln_bias_main_grad
        ),
        stream
    ], DCFunc.ln_mlp_bwd

def ln_(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    bias = torch.randn(h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    input_args = [config,(input,weight,bias),stream]
    return input_args, DCFunc.ln_attn

def gemm_attn_c(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(3*h//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_attn_column

def gemm_attn_c_wgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,
    device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(3*h//tp,h,dtype=torch.float16,
    device=f"cuda:{torch.cuda.current_device()}")
    dgrad = torch.randn(b*s,3*h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,dgrad,weight),stream], DCFunc.gemm_wgrad_attn_c

def gemm_attn_c_dgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,3*h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(3*h//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_dgrad_attn_c

def gemm_attn_l(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_attn_linear

def gemm_attn_l_wgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    grad = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    # input = torch.transpose(input,dim0=0,dim1=1)
    return [config,(grad,input,weight),stream], DCFunc.gemm_wgrad_attn_l

def gemm_attn_l_dgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,h//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_dgrad_attn_l

def gemm_mlp_c(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(2*ffn//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_mlp_column

def gemm_mlp_c_wgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    dgrad = torch.randn(b*s,2*ffn//tp,dtype=torch.float16,
    device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(2*ffn//tp,h,dtype=torch.float16,
    device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,dgrad,weight),stream], DCFunc.gemm_wgrad_mlp_c

def gemm_mlp_c_dgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,2*ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(2*ffn//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    output = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_dgrad_mlp_c

def gemm_mlp_l(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    output = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    # print_rank0(f"input:{input.size()} weight:{weight.size()}")
    return [config,(input,weight),stream], DCFunc.gemm_mlp_linear

def gemm_mlp_l_wgrad(b,s,h,a,tp,ffn,stream):
    dgrad = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    input = torch.randn(b*s,ffn//tp,dtype=torch.float16,device="cuda")
    output = torch.randn(h,ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(dgrad,input,weight),stream], DCFunc.gemm_wgrad_mlp_l

def gemm_mlp_l_dgrad(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    weight = torch.randn(h,ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    output = torch.randn(b*s,ffn//tp,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,weight),stream], DCFunc.gemm_dgrad_mlp_l

def reduce_scatter(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,True),stream],DCFunc.reduce_scatter

def allgather(b,s,h,a,tp,ffn,stream):
    input = torch.randn(b*s//tp,h,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}")
    return [config,(input,True),stream],DCFunc.allgather

OP_TABLE_FWD = {
    "ln_attn":ln_,
    "allgather_attn":allgather,
    "gemm_attn_c":gemm_attn_c,
    "fl_attn":fa_,
    "cp_attn_block_fwd_1": cp_fa_,
    "cp_attn_block_fwd_2": cp_fa_last,
    "gemm_attn_l":gemm_attn_l,
    "reduce_scatter_attn": reduce_scatter,
    "bda_attn":bda_,
    "ln_mlp":ln_,
    "allgather_mlp":allgather,
    "gemm_mlp_c":gemm_mlp_c,
    "swiglu_mlp":swiglu_,
    "gemm_mlp_l":gemm_mlp_l,
    "reduce_scatter_mlp": reduce_scatter,
    "bda_mlp":bda_,
}
OP_TABLE_BWD = {
    "bda_mlp_bwd":bda_bwd_,
    "allgather_mlp_bwd":allgather,
    "gemm_dgrad_mlp_l":gemm_mlp_l_dgrad,
    "gemm_wgrad_mlp_l":gemm_mlp_l_wgrad,
    "swiglu_bwd":swiglu_bwd,
    "gemm_dgrad_mlp_c":gemm_mlp_c_dgrad,
    "gemm_wgrad_mlp_c":gemm_mlp_c_wgrad,
    "reduce_scatter_mlp_c_bwd":reduce_scatter,
    "ln_mlp_bwd":ln_bwd,
    "bda_attn_bwd":bda_bwd_,
    "allgather_attn_bwd":allgather,
    "gemm_dgrad_attn_l":gemm_attn_l_dgrad,
    "gemm_wgrad_attn_l":gemm_attn_l_wgrad,
    "cp_attn_block_bwd_1" : cp_fa_bwd,
    "cp_attn_block_bwd_2" : cp_fa_bwd,
    "cp_attn_block_bwd_3" : cp_fa_bwd_last,
    "attn_bwd":fa_bwd,
    "gemm_dgrad_attn_c":gemm_attn_c_dgrad,
    "gemm_wgrad_attn_c":gemm_attn_c_wgrad,
    "reduce_scatter_attn_c_bwd":reduce_scatter,
    "ln_attn_bwd":ln_bwd,
}

def get_candidate_operator_groups():
    operator_list_prune(context_parallel_size=args.context_parallel_size)
    REVERSE_OP_TABLE_INDEX = {value: key for key, value in OP_TABLE_IDX.items()}
    def generate_candidate_operator_groups(operators_list):
        candidate_operator_groups = [[]]
        for sub_list in operators_list:
            ctx_len = 1
            while ctx_len <= len(sub_list):
                for start in range(len(sub_list)-ctx_len+1):
                    end = start + ctx_len
                    candidate_operator_groups.append([REVERSE_OP_TABLE_INDEX[op_id] for op_id in sub_list[start:end]])
                ctx_len = ctx_len+1
        return candidate_operator_groups
    candidate_operator_groups_fwd=generate_candidate_operator_groups(fwd_operators_list)
    candidate_operator_groups_bwd=generate_candidate_operator_groups(bwd_operators_list)
    return candidate_operator_groups_fwd, candidate_operator_groups_bwd


