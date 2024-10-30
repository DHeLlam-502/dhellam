from torch import Tensor
from typing import Any

def pygemm(inputA: Tensor, 
           inputB: Tensor, 
           outputD: Tensor, 
           input_workspace: Tensor, 
           workspaceSize: int, 
           transa: bool, 
           transb: bool, 
           grad_accumulate: bool, 
           math_sm_count: int, 
           profiling: bool) -> float: ...

def pygelu(input: Tensor, 
           output: Tensor, 
           profiling: bool) -> float: ...