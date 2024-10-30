import torch
import time
from dhellam.common.common import print_rank0
def benchmark(repeat, fn, *args, **kwargs):
    time_list = list()
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        elapsed_time = (end-start)*1000
        time_list.append(elapsed_time)
    return sum(time_list)/repeat


def benchmark_ms(repeat, s1, s2, fn1, fn2, *args):
    fn1_args, kfn1 = fn1(*args)
    fn2_args, kfn2 = fn2(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        with torch.cuda.stream(s1):
            kfn1(*fn1_args)
        with torch.cuda.stream(s2):
            kfn2(*fn2_args)
    torch.cuda.synchronize()
    end = time.time()
    return (end-start)/repeat*1000

def benchmark_ms_ops(repeat, s1, s2, fn1, fn2, fn3, *args):

    stream1_list = []
    stream2_list = []
    for op in fn1:
        fn1_args, kfn1 = op(*args,s1)
        tmp = [fn1_args, kfn1]
        stream1_list.append(tmp)
    for op in fn2:
        fn2_args, kfn2 = op(*args,s2)
        tmp = [fn2_args, kfn2]
        stream2_list.append(tmp)
    if fn3 is not None:
        fn3_args,kfn3 = fn3(*args,s1)
    all_time =0
    time_list = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        if fn3 is not None:
            if s1 == s2:
                input,async_op = fn3_args[1]
                _ , handle = kfn3(fn3_args[0],(input,False),fn3_args[2])
                if fn1 is not None or fn2 is not None:
                    torch.cuda.synchronize()
            else:
                _ , handle = kfn3(*fn3_args)
        
        for item in stream1_list:
            item[1](*item[0])
        
        for item in stream2_list:
            item[1](*item[0])
        torch.cuda.synchronize()
        tmp =  time.time()-start
        if torch.distributed.get_rank()==0:
            time_list.append(tmp)
    if torch.distributed.get_rank()==0:
        time_list.sort()
    total_items = len(time_list)
    percentage_to_remove = 0.2
    items_to_remove = int(total_items * percentage_to_remove)
    if round(items_to_remove,0) > 0 and total_items>10:
            time_list = time_list[items_to_remove:-items_to_remove]
    if len(time_list)>0:
        origin = sum(value for  value in time_list) / len(time_list)
    else:
        origin = 0
    return origin*1000
    

def benchmark_ms_ops_cp(repeat, s1, s2, fn1, fn2, fn3, wait_fwd,wait_bwd,parallel,*args):

    stream1_list = []
    stream2_list = []
    stream3_list = []
    for op in fn1:
        fn1_args, kfn1 = op(*args,s1)
        tmp = [fn1_args, kfn1]
        stream1_list.append(tmp)
    for op in fn2:
        fn2_args, kfn2 = op(*args,s2)
        tmp = [fn2_args, kfn2]
        stream2_list.append(tmp)
    for op in fn3:
        fn3_args,kfn3 = op(*args,s1)
        tmp = [fn3_args,kfn3]
        stream3_list.append(tmp)
    time_list = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        handler_list = []
        for item in stream3_list:
            if not parallel:
                fn3_args = item[0]
                _ , handle = item[1](*fn3_args)
                handler_list.append(handle)
                torch.cuda.synchronize()
            else:
                _ , handle = item[1](*item[0])
                handler_list.append(handle)
        
        if wait_fwd:
            for item in stream2_list:
                item[1](*item[0])
            
            handler_list[0].wait()
            for item in stream1_list:
                item[1](*item[0])
        elif wait_bwd:
            for item in stream1_list:
                item[1](*item[0])
            
            handler_list[0].wait()
            for item in stream2_list:
                item[1](*item[0])
        else:
            for item in stream1_list:
                item[1](*item[0])
            for item in stream2_list:
                item[1](*item[0])

        torch.cuda.synchronize()
        tmp =  time.time()-start
        if torch.distributed.get_rank()==0:
            time_list.append(tmp)
    if torch.distributed.get_rank()==0:
        time_list.sort()
    total_items = len(time_list)
    percentage_to_remove = 0.2 
    items_to_remove = int(total_items * percentage_to_remove)
    if round(items_to_remove,0) > 0 and total_items>10:
            time_list = time_list[items_to_remove:-items_to_remove]
    if len(time_list)>0:
        origin = sum(value for  value in time_list) / len(time_list)
    else:
        origin = 0
    return origin*1000

def benchmark_dhellam(repeat, s1, s2, fn1, fn2, fn3, *args):
    stream1_list = []
    stream2_list = []
    for op in fn1:
        fn1_args, kfn1 = op(*args)
        tmp = [fn1_args, kfn1]
        stream1_list.append(tmp)
    for op in fn2:
        fn2_args, kfn2 = op(*args)
        tmp = [fn2_args, kfn2]
        stream2_list.append(tmp)
    if fn3 is not None:
        fn3_args,kfn3 = fn3(*args)
    all_time =0
    time_list = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        if fn3 is not None:
            if s1 == s2:
                _ , handle = kfn3(fn3_args[0],fn3_args[1],None,False,fn3_args[4])
                torch.cuda.synchronize()
            else:
                _ , handle = kfn3(*fn3_args)
        with torch.cuda.stream(s1):
            for item in stream1_list:
                item[1](*item[0])
        with torch.cuda.stream(s2):
            for item in stream2_list:
                item[1](*item[0])
        torch.cuda.synchronize()
        tmp =  time.time()-start
        if torch.distributed.get_rank()==0:
            time_list.append(tmp)
    if torch.distributed.get_rank()==0:
        time_list.sort()
    total_items = len(time_list)
    percentage_to_remove = 0.2 
    items_to_remove = int(total_items * percentage_to_remove)
    if round(items_to_remove,0) > 0 and total_items>10:
            time_list = time_list[items_to_remove:-items_to_remove]
    if len(time_list)>0:
        origin = sum(value for  value in time_list) / len(time_list)
    else:
        origin = 0
    return origin*1000