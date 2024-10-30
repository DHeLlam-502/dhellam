import torch
import torch.distributed
from dhellam.common.benchmark import benchmark_ms_ops, benchmark_ms_ops_cp
from dhellam.common.common import print_rank0
from collections import defaultdict
from dhellam.adaptor.megatron import initialize_dhellam_megatron
import os
from tqdm import tqdm
from contextlib import nullcontext
args = initialize_dhellam_megatron(os.getenv('MegaPath',None))
from op_table import OP_TABLE_IDX,OP_TABLE_FWD,OP_TABLE_BWD,get_candidate_operator_groups

B = args.micro_batch_size
S = args.seq_length
H = args.hidden_size
A = args.num_attention_heads
TP = args.tensor_model_parallel_size
FFN = args.ffn_hidden_size

def test(fn1_array, fn2_array):
    speedup=0
    s1 = torch.cuda.default_stream()
    s2 = torch.cuda.Stream()
    fn1=[]
    fn2=[]
    fn3 =None
    for fn1_op in fn1_array:
        if "reduce_scatter" not in fn1_op and "allgather" not in fn1_op:
            fn1.append(OP_TABLE_FWD[fn1_op])
        else:
            fn3 = OP_TABLE_FWD[fn1_op]
    for fn2_op in fn2_array:
        if fn2_op == "gemm_dgrad_attn_c_allgather_attn_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_dgrad_attn_c"])
            fn3 = OP_TABLE_BWD["allgather_attn_c_bwd"]
        elif fn2_op == "gemm_wgrad_attn_c_reduce_scatter_attn_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_wgrad_attn_c"])
            fn3 = OP_TABLE_BWD["reduce_scatter_attn_c_bwd"]
        elif fn2_op == "gemm_wgrad_mlp_c_reduce_scatter_mlp_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_wgrad_mlp_c"])
            fn3 = OP_TABLE_BWD["reduce_scatter_mlp_c_bwd"]
        elif fn2_op == "gemm_dgrad_mlp_c_allgather_mlp_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_dgrad_mlp_c"])
            fn3 = OP_TABLE_BWD["allgather_mlp_c_bwd"]
        elif "reduce_scatter" not in fn2_op and "allgather" not in fn2_op:
            fn2.append(OP_TABLE_BWD[fn2_op])
        else:
            fn3 = OP_TABLE_BWD[fn2_op]
    benchmark_ms_ops(1,s1,s2,fn1,fn2,fn3,B,S,H,A,TP,FFN)
    benchmark_ms_ops(1,s1,s1,fn1,fn2,fn3,B,S,H,A,TP,FFN)
    benchmark_ms_ops(1,s2,s2,fn1,fn2,fn3,B,S,H,A,TP,FFN)
    seq_time =  benchmark_ms_ops(5,s1,s1,fn1,fn2,fn3,B,S,H,A,TP,FFN)
    para_time = benchmark_ms_ops(5,s1,s2,fn1,fn2,fn3,B,S,H,A,TP,FFN)
    if torch.distributed.get_rank()==0:
        speedup = seq_time/para_time
    else:
        speedup = 1
    fn1_array_num = []
    fn2_array_num = []
    for fn1_arrary_op in fn1_array:
        fn1_array_num.append(OP_TABLE_IDX[fn1_arrary_op])
    
    for fn2_array_op in fn2_array:
        fn2_array_num.append(OP_TABLE_IDX[fn2_array_op])
    return fn1_array_num, fn2_array_num, speedup, seq_time, para_time

def test_cp(fn1_array, fn2_array):
    speedup=0
    s1 = torch.cuda.default_stream()
    s2 = torch.cuda.Stream()
    s2 = s1 
    fn1 = []
    fn2 = []
    fn3 = []
    fn3_str = []  #* fn3 should be a list from now
    need_wait_fwd = False
    need_wait_bwd = False  #* may need to wait due to mix combination of TP op and Comp op

    for fn1_op in fn1_array:
        if "reduce_scatter" in fn1_op or "allgather" in fn1_op:
            fn3_str.append(fn1_op)
            need_wait_fwd =True
        elif ("cp_attn" in fn1_op and int(fn1_op[-1])!= args.context_parallel_size):
            fn3_str.append(fn1_op)
            break
        else:
            fn1.append(OP_TABLE_FWD[fn1_op])

    for fn2_op in fn2_array:
        if fn2_op == "gemm_dgrad_attn_c_allgather_attn_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_dgrad_attn_c"])
            fn3_str.append("allgather_attn_c_bwd")
        elif fn2_op == "gemm_wgrad_attn_c_reduce_scatter_attn_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_wgrad_attn_c"])
            fn3_str.append("reduce_scatter_attn_c_bwd")
        elif fn2_op == "gemm_wgrad_mlp_c_reduce_scatter_mlp_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_wgrad_mlp_c"])
            fn3_str.append("reduce_scatter_mlp_c_bwd")
        elif fn2_op == "gemm_dgrad_mlp_c_allgather_mlp_c_bwd":
            fn2.append(OP_TABLE_BWD["gemm_dgrad_mlp_c"])
            fn3_str.append("allgather_mlp_c_bwd")
        elif "reduce_scatter"  in fn2_op or "allgather"  in fn2_op:
            fn3_str.append(fn2_op)
            need_wait_bwd = True
        elif ("cp_attn" in fn2_op and int(fn2_op[-1])!= args.context_parallel_size+1):
            fn3_str.append(fn2_op)
            break
        else:
            fn2.append(OP_TABLE_BWD[fn2_op])

    cp_comms = [ comm_op for comm_op in fn3_str if "cp_attn" in comm_op]
    tp_comms = [ comm_op for comm_op in fn3_str if "cp_attn" not in comm_op]
    fn3_str = tp_comms+cp_comms

    for comm_op in fn3_str:
        if "bwd" in comm_op:
            fn3.append(OP_TABLE_BWD[comm_op])
        else:
            fn3.append(OP_TABLE_FWD[comm_op])
    
    benchmark_ms_ops_cp(1,s1,s2,fn1,fn2,fn3,need_wait_fwd,need_wait_bwd,True,B,S,H,A,TP,FFN)
    benchmark_ms_ops_cp(1,s1,s1,fn1,fn2,fn3,need_wait_fwd,need_wait_bwd,False,B,S,H,A,TP,FFN)
    benchmark_ms_ops_cp(1,s2,s2,fn1,fn2,fn3,need_wait_fwd,need_wait_bwd,False,B,S,H,A,TP,FFN)
    seq_time =  benchmark_ms_ops_cp(1,s1,s1,fn1,fn2,fn3,need_wait_fwd,need_wait_bwd,False,B,S,H,A,TP,FFN)
    # time.sleep(0.5)
    para_time = benchmark_ms_ops_cp(1,s1,s2,fn1,fn2,fn3,need_wait_fwd,need_wait_bwd,True,B,S,H,A,TP,FFN)
    if torch.distributed.get_rank()==0:
        speedup = seq_time/para_time
    else:
        speedup = 1

    fn1_array_num = []
    fn2_array_num = []
    for fn1_arrary_op in fn1_array:
        fn1_array_num.append(OP_TABLE_IDX[fn1_arrary_op])
    
    for fn2_array_op in fn2_array:
        fn2_array_num.append(OP_TABLE_IDX[fn2_array_op])
    return fn1_array_num, fn2_array_num, speedup, seq_time, para_time


if __name__ == "__main__" :
    fwd_operators, bwd_operators = get_candidate_operator_groups()
    result_table = defaultdict(lambda: {})

    progress_bar = tqdm(total=len(fwd_operators)*len(bwd_operators)) if torch.distributed.get_rank()==0 else nullcontext()
    with progress_bar as pbar:
        for fwd_ops in fwd_operators:
            for bwd_ops in bwd_operators:

                if len(fwd_ops)>0 and len(bwd_ops)>0:
                    fwd_comm = 0
                    bwd_comm = 0
                    for op in fwd_ops:
                        if "allgather" in op or "reduce_scatter" in op:
                            fwd_comm = 1
                            break
                        elif ("cp_attn" in op and int(op[-1])!= args.context_parallel_size):
                            fwd_comm =2
                            break
                    for op in bwd_ops:
                        if "allgather" in op or "reduce_scatter" in op:
                            bwd_comm = 1
                            break
                        elif ("cp_attn" in op and int(op[-1])!= args.context_parallel_size+1):
                            bwd_comm =2
                            break
                    if ((fwd_comm == bwd_comm) and (fwd_comm>0)) :
                        continue
                if args.context_parallel_size == 1:
                    (fn1_array_num, fn2_array_num, speedup, seq_time, para_time) = test(fwd_ops,bwd_ops)
                elif args.context_parallel_size == 2:
                    (fn1_array_num, fn2_array_num, speedup, seq_time, para_time) = test_cp(fwd_ops,bwd_ops)
                else:
                    print_rank0(f"Context Parallel Size Error: {args.context_parallel_size}")
                    exit(-1)
                if torch.distributed.get_rank()==0:
                    pbar.update(1)
                    with open('op_profile_num.txt','a') as fp:
                        fp.write(f"[{fn1_array_num},{fn2_array_num}] speedup: {speedup} seq_time:{seq_time} para_time:{para_time}\n")



