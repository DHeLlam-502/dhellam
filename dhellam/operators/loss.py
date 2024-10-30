import torch
from dhellam.common.template_launch import *
from dhellam.adaptor.parallel_state import *
from typing import List, Sequence
import os


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank, world_size: int
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l

def cross_entropy_fwd(inputs,label_smoothing=0.0):
    
    vocab_parallel_logits = inputs["vocab_parallel_logits"]
    target = inputs["target"]

    # Maximum value along vocab dimension across all GPUs.
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    torch.distributed.all_reduce( 
        logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
    )
    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

    # Get the partition's vocab indecies
    get_vocab_range = vocab_range_from_per_partition_vocab_size
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0

    # Get predicted-logits = logits[target].
    # For Simplicity, we convert logits to a 2-D tensor with size
    # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d] 
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits[target_mask] = 0.0 
    # All reduce is needed to get the chunks from other GPUs.
    torch.distributed.all_reduce( 
        predicted_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )

    # Sum of exponential of logits along vocab dimension across all GPUs.
    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )
    loss = torch.log(sum_exp_logits) - predicted_logits

    # Normalize and optionally smooth logits
    exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

    vocab_size = exp_logits.size(-1)
    if label_smoothing > 0:
        """
        We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
        = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
        = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
        = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
        = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
        = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
        From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
        """
        assert 1.0 > label_smoothing > 0.0
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)

        # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
        log_probs = torch.log(exp_logits)
        mean_log_probs = log_probs.mean(dim=-1)
        loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

    output_dict={}
    output_dict["loss"]= loss
    output_dict["label_smoothing"]=label_smoothing
    output_dict["vocab_size"] = vocab_size
    output_dict["softmax"] =exp_logits
    output_dict["target_mask"] = target_mask
    output_dict["masked_target_1d"] =masked_target_1d

    return output_dict

def cross_entropy_bwd(context,dout):
    input_dict = context[0]
    output_dict = context[1]

    softmax, target_mask, masked_target_1d = output_dict["softmax"],output_dict["target_mask"],output_dict["masked_target_1d"]
    label_smoothing, vocab_size = output_dict["label_smoothing"], output_dict["vocab_size"]

    # All the inputs have softmax as thier gradient.
    grad_input = softmax
    # For simplicity, work with the 2D gradient.
    partition_vocab_size = softmax.size()[-1]
    grad_2d = grad_input.view(-1, partition_vocab_size)

    # Add the gradient from matching classes.
    arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
    softmax_update = 1.0 - target_mask.view(-1).float()

    if label_smoothing > 0:
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)
        grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
        average_grad = 1 / vocab_size
        grad_2d[arange_1d, :] -= smoothing * average_grad
    else:
        grad_2d[arange_1d, masked_target_1d] -= softmax_update

    # Finally elementwise multiplication with the output gradients.
    grad_input.mul_(dout.unsqueeze(dim=-1))
    return grad_input



def mask_scale_loss_fwd(inputs):

    loss_mask = inputs["loss_mask"]
    output_tensor = inputs["output_tensor"]
    num_mb = inputs["num_microbatch"]
    scale_value = inputs["loss_scale"]

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    
    # Check individual rank losses are not NaN
    global_rank = torch.distributed.get_rank()
    assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    loss = (loss/num_mb)*scale_value

    output_dict={}
    output_dict["loss"] = loss
    return output_dict
    
def mask_scale_loss_bwd(context,dout):
    if dout is not None:
        dout = None
    input_dict = context[0]
    output_dict = context[1]
    final_loss = output_dict["loss"]
    output_tensor = input_dict["output_tensor"]

    torch.autograd.backward(final_loss,grad_tensors=dout,inputs=[output_tensor],retain_graph=True)
    return output_tensor.grad
