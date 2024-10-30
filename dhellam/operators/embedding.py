import torch
from torch.nn.parameter import Parameter
from dhellam.common.template_launch import *
from dhellam.adaptor.parallel_state import *
from typing import List, Sequence


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank, world_size: int
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def word_embedding_fwd(inputs):
    vocab=inputs["vocab"]
    token_ids=inputs["token_ids"]

    tp_vocab_size = vocab.shape[0]
    (vocab_start_index,
     vocab_end_index) = vocab_range_from_per_partition_vocab_size(tp_vocab_size,get_tensor_model_parallel_rank(),get_tensor_model_parallel_world_size())
    if get_tensor_model_parallel_world_size() > 1:
        # Build the mask.
        input_mask = (token_ids < vocab_start_index) | (token_ids >= vocab_end_index)
        # Mask the input.
        masked_input = token_ids.clone() - vocab_start_index
        masked_input[input_mask] = 0
    else:
        masked_input = token_ids
    # Get the embeddings.
    output_parallel = vocab[masked_input]
    # Mask the output embedding.
    if get_tensor_model_parallel_world_size() > 1:
        output_parallel[input_mask, :] = 0.0

    output_dict={}
    output_dict["word_embeddings"]=output_parallel
    return output_dict


def word_embedding_bwd(context,dout):
    input_dict = context[0]
    output_dict = context[1]
    out=output_dict["word_embeddings"]
    vocab=input_dict["vocab"]
    token_ids=input_dict["token_ids"]
    torch.autograd.backward(out,grad_tensors=dout,inputs=[vocab],retain_graph=True)
    return None

def pos_embedding():
    pass

