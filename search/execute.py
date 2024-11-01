import os
from dhellam.adaptor.megatron import initialize_dhellam_megatron
args = initialize_dhellam_megatron(os.getenv('MegaPath',None))
import torch
import time
from dhellam.common.common import print_rank0
from dhellam.core.dstb import DSTB, DCoperation, exec
import shutil
from megatron import get_args
from megatron.arguments import core_transformer_config_from_args
from datetime import datetime
from dhellam.common.global_variables import GLOBAL_VARIABLES

config = core_transformer_config_from_args(get_args())

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_data_parallel_group,
    get_context_parallel_group,
    get_context_parallel_global_ranks,
    get_context_parallel_rank
)

if __name__ == "__main__":
    configs_path = f'config/dhellam-cp{args.context_parallel_size}/'
    result = {}
    max_speedup = -100000
    max_file = None
    print_rank0(configs_path)

    all_files = [(root,file) for root, dirs, files in os.walk(configs_path) for file in files if file.endswith('.json')]
    all_files = [all_files[i] for i in range(get_data_parallel_rank(), len(all_files), get_data_parallel_world_size())]

    stream1 = torch.cuda.default_stream()
    stream2 = torch.cuda.Stream()
    if config.context_parallel_size > 1:
        stream2 = stream1   #* 对CP来说这个是必要的

    fwd_input = torch.randn(args.micro_batch_size, args.seq_length//(config.tensor_model_parallel_size*config.context_parallel_size), args.hidden_size,dtype=torch.float16,device="cuda")
    bwd_input = torch.randn(args.micro_batch_size, args.seq_length//(config.tensor_model_parallel_size*config.context_parallel_size), args.hidden_size,dtype=torch.float16,device="cuda")
    config.tp_group = get_tensor_model_parallel_group(check_initialized=False)
    config.tp_size = config.tensor_model_parallel_size
    config.cp_group = get_context_parallel_group(check_initialized=False)
    config.cp_global_ranks = get_context_parallel_global_ranks(check_initialized=False)
    config.seq_length = args.seq_length//args.context_parallel_size


    sequential_time = None
    times = 20

    with torch.no_grad():
        for i,(root, file) in enumerate(all_files):
            print_rank0(f">>>>now: {file}")
            args.schedule_config = os.path.join(root,file)
            activation_dict = {}
            new_activation_dict = {}
            
            dctb_test = DSTB(config, stream1, stream2).cuda().half()
            for param in dctb_test.parameters():
                param.main_grad = torch.zeros_like(param)

            # warmup
            torch.cuda.synchronize()
            exec(config,DCoperation,dctb_test.weight_second,None,fwd_input,None,dctb_test.fwd_only_dcblocks,stream1,stream2,{},activation_dict)
            exec(config,DCoperation,None,dctb_test.weight_first,None,bwd_input,dctb_test.bwd_only_dcblocks,stream1,stream2,activation_dict,{})
            torch.cuda.synchronize()

            if sequential_time == None:
                time_list = []
                for _ in range(times):
                    torch.distributed.barrier(config.tp_group)
                    torch.cuda.synchronize()
                    start = time.time()
                    exec(config,DCoperation,dctb_test.weight_second,None,fwd_input,None,dctb_test.fwd_only_dcblocks,stream1,stream2,{},activation_dict)
                    exec(config,DCoperation,None,dctb_test.weight_first,None,bwd_input,dctb_test.bwd_only_dcblocks,stream1,stream2,activation_dict,{})
                    torch.cuda.synchronize()
                    end = time.time()
                    time_list.append((end-start)*1000)
                sequential_time = sum(time_list)/len(time_list)
            for param in dctb_test.parameters():
                param.main_grad = torch.zeros_like(param)

            exec(config,DCoperation,dctb_test.weight_second,None,fwd_input,None,dctb_test.fwd_only_dcblocks,stream1,stream2,{},activation_dict)
            exec(config,DCoperation,dctb_test.weight_second,dctb_test.weight_first,fwd_input,bwd_input,dctb_test.dcblocks,stream1,stream2,activation_dict,new_activation_dict)
            activation_dict = new_activation_dict
            new_activation_dict = {}
            time_list = []
            torch.cuda.synchronize()
            for _ in range(times):
                torch.distributed.barrier(config.tp_group)
                torch.cuda.synchronize()
                start = time.time()
                exec(config,DCoperation,dctb_test.weight_second,dctb_test.weight_first,fwd_input,bwd_input,dctb_test.dcblocks,stream1,stream2,activation_dict,new_activation_dict)
                torch.cuda.synchronize()
                end = time.time()
                activation_dict = new_activation_dict
                new_activation_dict = {}
                time_list.append((end-start)*1000)
            dctb_time = sum(time_list)/len(time_list)
            speedup = ((sequential_time/dctb_time)-1)*100
            if speedup > max_speedup:
                max_speedup = speedup
                max_file = file
            print_rank0(f"[{i+1}/{len(all_files)}] sequence time :{sequential_time:.2f} schedule time: {dctb_time:.2f} speedup/max: {speedup:.2f}/{max_speedup:.2f} file/max: {file}({max_file})")
            GLOBAL_VARIABLES.clear()
    max_speedup_tensor = torch.tensor(max_speedup,dtype=torch.float64,device="cuda")
    if get_tensor_model_parallel_rank()==0 and get_context_parallel_rank() ==0 :
        torch.distributed.all_reduce(max_speedup_tensor,op=torch.distributed.ReduceOp.MAX,group=get_data_parallel_group())
        if max_speedup==max_speedup_tensor.item():
            print(f"[rank-{torch.distributed.get_rank()}] summary report: max speedup is {max_speedup_tensor.item()} file: {max_file}")
            if os.path.exists(os.path.join(configs_path,max_file)):
                print(f"copy max_file to config/best.json")
                shutil.copy(src=os.path.join(configs_path,max_file), dst=f'config/best.json')
    torch.distributed.barrier()
    if torch.distributed.get_rank()==0:
        dp_file = os.path.join(configs_path, 'dp_best.json')
        rename_file = f'dp_best_{datetime.now().time()}.json'
        target_path = os.path.join(configs_path, rename_file)
        if os.path.exists(dp_file) and max_file == 'dp_best.json':
            print_rank0(f"rename {max_file} to {rename_file}")
            os.rename(dp_file,target_path)