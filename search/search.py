import re
from functools import lru_cache
import sys,os
import ast
import json
from op_table import OP_TABLE_IDX, operator_list_prune

fwd_operators = None
bwd_operators = None
context_parallel_size = int(sys.argv[1])
result_dict = {}

def illegal_check(fwd_list_str, bwd_list_str):
    situation1 = ((any('allgather' in name for name in fwd_list_str) or any('reduce' in name for name in fwd_list_str)) and \
               (any('allgather' in name for name in bwd_list_str) or any('reduce' in name for name in bwd_list_str)))
    
    situation2 =  any( ('cp_attn' in name  and int(name[-1])!=context_parallel_size)  for name in fwd_list_str) and \
                  any( ('cp_attn' in name  and int(name[-1])!=context_parallel_size+1)  for name in bwd_list_str)

    illgeal = situation1 or situation2
    return illgeal

def sort_sublist_and_inner_elements(unsorted_list, order_map):
    sorted_list = sorted(unsorted_list, key=lambda x: min(order_map[item] for item in x if item in order_map))
    sorted_list = [sorted(sublist, key=lambda x: order_map[x] if x in order_map else float('inf')) for sublist in sorted_list]
    return sorted_list

def generate_sub_groups(ops_list, n):
    @lru_cache(maxsize=None)
    def split_list(input_list):
        input_list = tuple(input_list)
        result = []
        length = len(input_list)
        for i in range(1, length):
            part1 = input_list[:i]
            part2 = input_list[i:]
            result.append((part1, part2))
        return result

    def generate_splits(lists, n):
        if len(lists) == n:
            return {frozenset(tuple(frozenset(lst) for lst in lists))}
        
        all_splits = set()
        for index, lst in enumerate(lists):
            if len(lists) + 1 <= n:
                possible_splits = split_list(tuple(lst))
                for split in possible_splits:
                    new_lists = lists[:index] + [list(split[0]), list(split[1])] + lists[index+1:]
                    further_splits = generate_splits(new_lists, n)
                    all_splits.update(further_splits)
        
        return all_splits

    ops_groups = generate_splits(ops_list,n)
    result = []
    for way in ops_groups:
        result.append([list(sublist) for sublist in way])
    return result

def parse_op_file(file_path):
    pattern = r'\[\s*\[\s*((?:\d+(?:,\s*\d+)*)?)\s*\],\s*\[\s*((?:\d+(?:,\s*\d+)*)?)\s*\]\]\s*speedup:\s*([0-9.]+)\s*seq_time:\s*([0-9.]+)\s*para_time:\s*([0-9.]+)'
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                first_list = match.group(1)
                second_list = [int(num) for num in match.group(2).split(',') if num.strip()]
                para_time = float(match.group(5))
                key = f"[[{first_list}], {second_list}]"
                data_dict[key] = para_time
            else:
                print(f"line:{line} is not match")
    return data_dict

def dp(m,n):
    fwd_order_map = {key: i for i, key in enumerate([item for sublist in fwd_operators for item in sublist])}
    bwd_order_map = {key: i for i, key in enumerate([item for sublist in bwd_operators for item in sublist])}
    fwd_result = generate_sub_groups(fwd_operators, m)
    for index,item in enumerate(fwd_result):
        fwd_result[index] = sort_sublist_and_inner_elements(item,fwd_order_map)
    bwd_result = generate_sub_groups(bwd_operators, n)
    for index,item in enumerate(bwd_result):
        bwd_result[index] = sort_sublist_and_inner_elements(item,bwd_order_map)
    tmp = sys.maxsize
    best_answer = None
    for fwd_item in fwd_result:
        for bwd_item in bwd_result:
            fwd_length = len(fwd_item)
            bwd_length = len(bwd_item)
            dhellam = [[0.0 for _ in range(bwd_length+1)] for _ in range(fwd_length+1)]
            dhellam_tmp = [[ [] for _ in range(bwd_length+1)] for _ in range(fwd_length+1)]
            for i in range(fwd_length+1):
                if i == 0:
                    dhellam[i][0] = 0.0
                else:
                    dhellam[i][0] = dhellam[i-1][0]+result_dict[f"[{fwd_item[i-1]}, []]"]
                    dhellam_tmp[i][0] = dhellam_tmp[i-1][0].copy() 
                    dhellam_tmp[i][0].append(f"[{fwd_item[i-1]}, []]")
            for i in range(bwd_length+1):
                if i == 0:
                    dhellam[0][i] = 0.0
                else:
                    dhellam[0][i] = dhellam[0][i-1]+result_dict[f"[[], {bwd_item[i-1]}]"]
                    dhellam_tmp[0][i] = dhellam_tmp[0][i-1].copy()  
                    dhellam_tmp[0][i].append(f"[[], {bwd_item[i-1]}]")
            id_to_op_name = {value: key for key, value in OP_TABLE_IDX.items()}
            for i in range(1,fwd_length+1):
                for j in range(1,bwd_length+1):
                    fwd_list = eval(f"{fwd_item[i-1]}")
                    bwd_list = eval(f"{bwd_item[j-1]}")
                    fwd_list_str = [id_to_op_name[num] for num in fwd_list]
                    
                    bwd_list_str = [id_to_op_name[num] for num in bwd_list]
                    if illegal_check(fwd_list_str, bwd_list_str): 
                        result = min(dhellam[i-1][j]+result_dict[f"[{fwd_item[i-1]}, []]"],
                                      dhellam[i][j-1]+result_dict[f"[[], {bwd_item[j-1]}]"])
                        dhellam[i][j]=result
                        if result == dhellam[i-1][j]+result_dict[f"[{fwd_item[i-1]}, []]"]:
                            dhellam_tmp[i][j] = dhellam_tmp[i-1][j].copy() 
                            dhellam_tmp[i][j].append(f"[{fwd_item[i-1]},[]]")
                        elif result == dhellam[i][j-1]+result_dict[f"[[], {bwd_item[j-1]}]"]:
                            dhellam_tmp[i][j] = dhellam_tmp[i][j-1].copy() 
                            dhellam_tmp[i][j].append(f"[[], {bwd_item[j-1]}]")
                    else:
                        result = min(dhellam[i-1][j]+result_dict[f"[{fwd_item[i-1]}, []]"],
                                        dhellam[i][j-1]+result_dict[f"[[], {bwd_item[j-1]}]"],
                                        dhellam[i-1][j-1]+result_dict[f"[{fwd_item[i-1]}, {bwd_item[j-1]}]"])
                        dhellam[i][j]=result
                        if result == dhellam[i-1][j]+result_dict[f"[{fwd_item[i-1]}, []]"]:
                            dhellam_tmp[i][j] = dhellam_tmp[i-1][j].copy() 
                            dhellam_tmp[i][j].append(f"[{fwd_item[i-1]},[]]")
                        elif result == dhellam[i][j-1]+result_dict[f"[[], {bwd_item[j-1]}]"]:
                            dhellam_tmp[i][j] = dhellam_tmp[i][j-1].copy() 
                            dhellam_tmp[i][j].append(f"[[], {bwd_item[j-1]}]")
                        else:
                            dhellam_tmp[i][j] = dhellam_tmp[i-1][j-1].copy() 
                            dhellam_tmp[i][j].append(f"[{fwd_item[i-1]}, {bwd_item[j-1]}]")
            if dhellam[i][j]<tmp:
                tmp = dhellam[i][j]
                best_answer = dhellam_tmp[i][j].copy() 
    return tmp,best_answer

if __name__ == "__main__":
    (fwd_min_group_num, fwd_max_group_num, fwd_operators),(bwd_min_group_num, bwd_max_group_num, bwd_operators) = operator_list_prune(context_parallel_size=context_parallel_size)

    print(f">>>> execute dp algo to find profiler best config (context parallel size: {context_parallel_size}) fwd_groups_num: ({fwd_min_group_num},{fwd_max_group_num}) bwd_groups_num: ({bwd_min_group_num},{bwd_max_group_num}) ...")
    file_path = 'op_profile_num.txt'  
    result_dict = parse_op_file(file_path)
    best_result = sys.maxsize
    best_answer = None
    for i in range(fwd_min_group_num, fwd_max_group_num+1):
        for j in range(bwd_min_group_num, bwd_max_group_num+1):
            result,answer = dp(i,j)
            if result<best_result:
                best_result = result
                best_answer = answer.copy()
    print(f">>>> dump the best answer to json file ...")
    def convert_numbers_to_names(data_list):
        op_names_table_inverse = {value: key for key, value in OP_TABLE_IDX.items()}
        return [[op_names_table_inverse.get(num, "Unknown") for num in sublist] for sublist in data_list]
    best_answer = [convert_numbers_to_names(ast.literal_eval(item)) for item in best_answer]

    def check_comm(op_list):
        check_result = False
        for op in op_list:
            if "reduce_scatter" in op or "allgather" in op:
                check_result = True
                break
            elif "cp_attn_block_fwd" in op and int(op[-1])!=context_parallel_size:
                check_result = True
                break
            elif "cp_attn_block_bwd" in op and int(op[-1])!=(context_parallel_size+1):
                check_result = True
                break
        return check_result
    
    # dump the best search result to file in json format
    json_structure = {"DCBlocks": []}
    for block in best_answer:
        third=[]
        first=[]
        second=[]
        if block[0] == []: 
            first = "None"
        elif check_comm(block[0]):
            if len(block[0])== 1:
                if block[0][0].count('_')==7:
                    underscore_positions = [i for i, char in enumerate(block[0][0]) if char == '_']
                    fourth_underscore_index = underscore_positions[3]
                    first.append(block[0][0][:fourth_underscore_index])
                    third.append(block[0][0][fourth_underscore_index + 1:])
                else:
                    third.append(block[0][0])
            else:
                if  block[0][0].count('_')==7:
                    underscore_positions = [i for i, char in enumerate(block[0][0]) if char == '_']
                    fourth_underscore_index = underscore_positions[3]
                    first.append(block[0][0][:fourth_underscore_index])
                    third.append(block[0][0][fourth_underscore_index + 1:])
                    first.extend(block[0][1:])
                else:
                    third.append(block[0][0])
                    first.extend(block[0][1:])                
        else:
            first = block[0]
        if block[1] == []:
            second = "None"
        elif  check_comm(block[1]):
            if len(block[1])== 1:
                if block[1][0].count('_')>=7:
                    underscore_positions = [i for i, char in enumerate(block[1][0]) if char == '_']
                    fourth_underscore_index = underscore_positions[3]
                    second.append(block[1][0][:fourth_underscore_index])
                    third.append(block[1][0][fourth_underscore_index + 1:])
                else:
                    third.append(block[1][0])
            else:
                if  block[1][0].count('_')>=7:
                    underscore_positions = [i for i, char in enumerate(block[1][0]) if char == '_']
                    fourth_underscore_index = underscore_positions[3]
                    second.append(block[1][0][:fourth_underscore_index])
                    third.append(block[1][0][fourth_underscore_index + 1:])
                    second.extend(block[1][1:])
                else:
                    third.append(block[1][0])
                    second.extend(block[1][1:])    
        else:
            second = block[1]
        if len(third) ==0:
            third = "None"
        else:
            cp_comms = [ comm_op for comm_op in third if "cp_attn" in comm_op]
            tp_comms = [ comm_op for comm_op in third if "cp_attn" not in comm_op]
            third = tp_comms+cp_comms
        json_structure['DCBlocks'].append([first, second, third])
    
    
    CONFIG_DIR = f'config/dhellam-cp{context_parallel_size}/'
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_DIR, 'best_result.json'), 'w') as json_file:
        json.dump(json_structure, json_file, indent=4)
