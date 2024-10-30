import torch
import torch.distributed
from megatron.core import tensor_parallel,parallel_state
from megatron import get_args
from dhellam.common.template_launch import template_fwd_new, template_bwd_new
from dhellam.core.dshb import DSHB
from dhellam.core.dstb import DSTB
from dhellam.common.global_variables import GLOBAL_VARIABLES
from dhellam.common.common import print_rank0
class Stack:
    def __init__(self):
        self.items = []

    def __str__(self) -> str:
        return f"dhellam.Stack object({len(self.items)} items)"

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")

    def size(self):
        return len(self.items)

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


class DSLLM(torch.nn.Module):
    def __init__(self, config, vocab_size, max_sequence_length, tp_group,cp_group,cp_global_ranks,pre_process=True) -> None:
        super(DSLLM, self).__init__()

        megatron_args = get_args()
        pp_stage_num = parallel_state.get_pipeline_model_parallel_world_size()
        assert config.num_layers%(2*pp_stage_num)==0, "Only support the llm with even number of layers in each pipeline stage!"
        num_layers_per_pp_stage = config.num_layers//(2*pp_stage_num)
        config.num_layers = num_layers_per_pp_stage
        self.config = config
        self.config.tp_group = tp_group
        self.config.cp_group = cp_group
        self.config.cp_global_ranks = cp_global_ranks
        self.config.tp_size = config.tensor_model_parallel_size
        self.config.vocab_size = vocab_size
        self.config.seq_length = megatron_args.seq_length//megatron_args.context_parallel_size
        stream1 = torch.cuda.default_stream()
        stream2 = torch.cuda.Stream()
        stream2 = stream1   #! temporarely use for following overlapping

        if pre_process:
            self.dchb = DSHB(config=config,stream1=stream1,stream2=stream1)
        self.dctb = torch.nn.Sequential(*[DSTB(config=config,stream1=stream1,stream2=stream2) for _ in range(num_layers_per_pp_stage)])
        self.pre_process = pre_process
        self.share_embeddings_and_output_weights = False  #* Don't need to share embedding weights

        #* required for pipeline parallel schedules
        self.input_tensor = None

        self.config.max_sequence_length = max_sequence_length
        self.register_full_backward_pre_hook(pre_backward_hook)

    def set_input_tensor(self, input_tensor: list ) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        
        if not self.pre_process:
            assert len(input_tensor) == 2, 'input_tensor should only be length 2 for DSLLM'
        self.input_tensor = input_tensor


    def forward(self, x1, x2): #* x1 stands for loss, x2 stands for input
        # only for testing
        if not self.pre_process:
            x1,x2 =  self.input_tensor

        backward_only = x2 is None
        forward_only = x1 is None

        assert not(forward_only and backward_only), "it's necessary to execute forward pass or backward pass"
        if not ('fwd_activations' in GLOBAL_VARIABLES) or not ('bwd_activations' in GLOBAL_VARIABLES):
            GLOBAL_VARIABLES['fwd_activations']=[] 
            GLOBAL_VARIABLES['bwd_activations']=[]
    
        if not backward_only:
            GLOBAL_VARIABLES['fwd_activations'].append(Stack())
        placeholder = torch.randn(1,dtype=torch.float16,device="cuda", requires_grad=True)
        GLOBAL_VARIABLES['inputs']=[x1,x2]
        if self.pre_process:
            placeholder = self.dchb(placeholder)
        placeholder = self.dctb(placeholder)  #TODO return value or not ?
        return placeholder


def pre_backward_hook(module, grad_output):
    x1, x2 = GLOBAL_VARIABLES["inputs"]
    backward_only = x2 is None
    forward_only = x1 is None  
    if not backward_only:
        GLOBAL_VARIABLES['bwd_activations'].append(Stack())

    return grad_output