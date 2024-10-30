import torch
import sys
import os
import math
from collections import defaultdict
import torch.distributed
from torch.nn.parameter import Parameter
from megatron.core.transformer import TransformerConfig
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron
from megatron import get_args
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group
from megatron.core import mpu
from cofutils import cofnsys
from dhellam.common.common import print_rank0
from dhellam.common.template_launch import template_fwd_new, template_bwd_new
from dhellam.common.global_variables import GLOBAL_VARIABLES
from dhellam import pylayernorm_fwd,pylayernorm_bwd


def loss_cal_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """    
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}

def initialize_pre_process_data(args):
    mbs = args.micro_batch_size
    seq_len =args.seq_length
    hidden_size = args.hidden_size

    vocab_size = args.padded_vocab_size
    tp_size = args.tensor_model_parallel_size
    sp_size = args.tensor_model_parallel_size
    tp_vocab_size = vocab_size//tp_size
    sp_seq_len = seq_len//sp_size
    
    token_ids = torch.randint(0,vocab_size,(mbs,seq_len),device=torch.cuda.current_device(),dtype=torch.int64)
    pos_ids = torch.randint(0,seq_len,(mbs,seq_len),device=torch.cuda.current_device(),dtype=torch.int64)
    hidden_states = torch.randn(sp_seq_len,mbs,hidden_size,device=torch.cuda.current_device(),dtype=torch.float16)
    hidden_states = hidden_states/hidden_states.max()
    hidden_states.requires_grad_()
    labels = torch.randint(0,vocab_size,(mbs,seq_len),device=torch.cuda.current_device(),dtype=torch.int64)
    loss_mask = torch.randint(0,vocab_size,(mbs,seq_len),device=torch.cuda.current_device(),dtype=torch.int64)>vocab_size//2
    loss_mask=loss_mask.to(torch.int64)

    return token_ids,pos_ids,hidden_states,labels,loss_mask

#? 
def pre_process(embedding,token_ids,pos_ids,embedding_weights):
    embed_result = embedding(token_ids,pos_ids)
    return embed_result

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,weight,bias,eps=1e-05,fwd_sm_margin=0,zero_centered_gamma=False,stream=None,profiling=False):
        output,mu,rsigma,_= pylayernorm_fwd(input,weight,bias,eps,fwd_sm_margin,zero_centered_gamma,stream,profiling)
        ctx.save_for_backward(input,weight,mu,rsigma)
        ctx.bwd_sm_margin = fwd_sm_margin
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.stream = stream
        ctx.profiling = profiling
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input,weight,mu,rsigma = ctx.saved_tensors
        dinp, dweight, dbias,elasped_time=pylayernorm_bwd(grad_output,input,mu,rsigma,weight,ctx.bwd_sm_margin,ctx.zero_centered_gamma,ctx.stream,ctx.profiling)
        return dinp,dweight,dbias,None,None,None,None,None


def post_process(output_layer, hidden_states, ln_weight,ln_bias,labels, loss_mask, word_embedding_weight,num_mb,loss_scale_func=None):

    with torch.enable_grad():
        hidden_states.requires_grad_(True)
        ln_fwd_output=LayerNormFunction.apply(hidden_states,ln_weight,ln_bias)
        # input: [s//tp,b,h]; output: [s,b,h]
        logits, _ = output_layer(ln_fwd_output, word_embedding_weight)
        # input: label([b,s]); output: loss([b,s])
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
        loss,_ = loss_cal_func(loss_mask=loss_mask, output_tensor=loss)
        GLOBAL_VARIABLES['loss'].append({'lm loss': loss.detach()})
        loss  = loss/num_mb
        if loss_scale_func is not None:
            loss = loss_scale_func(loss)
    return loss


class DSHB_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, embedding,output_layer,ln_weight,ln_bias,logits_weight,loss_func,num_mb,loss_scale_func=None,stream_fwd=None,stream_bwd=None):

        bwd_input, fwd_input = GLOBAL_VARIABLES['inputs']
        
        
        '''
        if fwd_input is not None:
            print("fwd input is not None:>")
        else:
            print("fwd input is None!")

        if bwd_input is not None:
            print("bwd input is not None:>")
        else:
            print("bwd input is None!")
        '''

        #* preparation for backward
        ctx.output_layer = output_layer
        ctx.ln_weight = ln_weight
        ctx.ln_bias = ln_bias
        ctx.logits_weight = logits_weight
        ctx.loss_func = loss_func
        ctx.num_mb = num_mb
        ctx.loss_scale_func = loss_scale_func
        ctx.stream_fwd = stream_fwd
        ctx.stream_bwd = stream_bwd
        
        if fwd_input is not None:
            token_ids,pos_ids,attention_mask,labels,loss_mask= fwd_input
            ctx.save_for_backward(labels,loss_mask)

        activation_list_new = list()
        if bwd_input is None:
            # only fwd
            with torch.enable_grad():
                embedding_ctx,_ = template_fwd_new(embedding, token_ids,pos_ids,stream=stream_fwd, profiling=False)
                activation_list_new.append(embedding_ctx)
                fwd_result = embedding_ctx[1]  #* Get the result of embedding
                with torch.cuda.stream(stream_fwd):
                    fwd_result = fwd_result.transpose(0,1).contiguous()
            
            GLOBAL_VARIABLES['fwd_activations'][-1].push(activation_list_new)
            GLOBAL_VARIABLES['inputs']=[None,fwd_result] #! Caution here
            return placeholder
        # get activations of the last micro-batch firstly
        act, = GLOBAL_VARIABLES['bwd_activations'][0].pop()  #* FILO 
        if GLOBAL_VARIABLES['bwd_activations'][0].is_empty():
            GLOBAL_VARIABLES['bwd_activations'].pop(0) 
        if fwd_input is None:
            # only bwd
            (dgrad,_,_,_),_ = template_bwd_new(act,None,stream=stream_bwd,profiling=False)
            with torch.cuda.stream(stream_bwd):
                dgrad = dgrad.transpose(0,1).contiguous()

            GLOBAL_VARIABLES['inputs']=[dgrad,None]
            return placeholder

        (dgrad,_,_,_),_ = template_bwd_new(act,None,stream=stream_bwd,profiling=False)
        act = None
        with torch.cuda.stream(stream_bwd):
            dgrad = dgrad.transpose(0,1).contiguous()
        # foward and backward pass here!
        with torch.enable_grad():
            embedding_ctx,_ = template_fwd_new(embedding, token_ids,pos_ids, stream=stream_fwd, profiling=False)
            activation_list_new.append(embedding_ctx)
            fwd_result = embedding_ctx[1]
            with torch.cuda.stream(stream_fwd):
                fwd_result = fwd_result.transpose(0,1).contiguous()
        GLOBAL_VARIABLES['fwd_activations'][-1].push(activation_list_new)
        GLOBAL_VARIABLES['inputs']=[dgrad,fwd_result]
        return placeholder


    @staticmethod
    def backward(ctx, *args):

        bwd_input, fwd_input = GLOBAL_VARIABLES['inputs']  #* input from DSTB
        '''
        if fwd_input is not None:
            print("The size of fwd input: ",fwd_input.shape)
        else:
            print("fwd input is None!")

        if bwd_input is not None:
            print("The size of bwd input: ",bwd_input.shape)
        else:
            print("bwd input is None!")
        '''
        if bwd_input is not None:
            with torch.cuda.stream(ctx.stream_bwd):
                bwd_input = bwd_input.transpose(0,1).contiguous()

        if fwd_input is not None:
            truth_labels, loss_mask = ctx.saved_tensors
            with torch.cuda.stream(ctx.stream_fwd):
                fwd_input = fwd_input.transpose(0,1).contiguous()

        activation_list_new = list()
        if bwd_input is None:
            # only fwd
            loss_ctx,_ = template_fwd_new(ctx.loss_func, ctx.output_layer,fwd_input,ctx.ln_weight,ctx.ln_bias,truth_labels,loss_mask, ctx.logits_weight,ctx.num_mb,ctx.loss_scale_func,stream=ctx.stream_fwd, profiling=False)
            activation_list_new.append(loss_ctx)
            fwd_result = loss_ctx[1]  #* Get the result of embedding

            GLOBAL_VARIABLES['bwd_activations'][-1].push(activation_list_new)
            GLOBAL_VARIABLES['inputs']=[None,fwd_result] #! Caution here
            return None,None,None,None,None,None,None,None,None,None,None
        # get activations of the last micro-batch firstly
        act, = GLOBAL_VARIABLES['fwd_activations'][0].pop()
        if GLOBAL_VARIABLES['fwd_activations'][0].is_empty():
            GLOBAL_VARIABLES['fwd_activations'].pop(0)
        if fwd_input is None:
            # only bwd
            template_bwd_new(act,bwd_input,stream=ctx.stream_bwd,profiling=False)
            GLOBAL_VARIABLES['inputs']=[None,None]
            return None,None,None,None,None,None,None,None,None,None,None
        # foward and backward pass here!
        template_bwd_new(act,bwd_input,stream=ctx.stream_bwd,profiling=False)
        loss_ctx,_ = template_fwd_new(ctx.loss_func, ctx.output_layer,fwd_input,ctx.ln_weight,ctx.ln_bias,truth_labels,loss_mask,ctx.logits_weight,ctx.num_mb,ctx.loss_scale_func,stream=ctx.stream_fwd, profiling=False)
        fwd_result = loss_ctx[1]

        activation_list_new.append(loss_ctx)
        GLOBAL_VARIABLES['bwd_activations'][-1].push(activation_list_new)
        GLOBAL_VARIABLES['inputs']=[None,fwd_result]
        return None,None,None,None,None,None,None,None,None,None,None

class DSHB(torch.nn.Module):
    def __init__(self, config, stream1=None, stream2=None) -> None:
        super(DSHB, self).__init__()
        megatron_args = get_args()
        self.stream_fwd = stream1
        self.stream_bwd = stream2
        self.embedding = LanguageModelEmbedding(
            config=config,
            vocab_size= config.vocab_size,
            max_sequence_length= megatron_args.max_position_embeddings,
            position_embedding_type='learned_absolute',
        ).cuda(torch.cuda.current_device()).half()
        self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output= False,
                skip_weight_param_allocation=True,
            ).cuda(torch.cuda.current_device()).half()
        self.ln_weight = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.ln_bias = Parameter(torch.empty(config.hidden_size,dtype=torch.float16,device=f"cuda:{torch.cuda.current_device()}"))
        self.logits_weight = self.embedding.word_embeddings.weight
        self.num_mb = math.ceil(megatron_args.global_batch_size/ megatron_args.micro_batch_size)
        self.loss_func = post_process

        setattr(self.ln_weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.ln_bias, 'sequence_parallel', config.sequence_parallel)

        GLOBAL_VARIABLES['loss']=[]
        #* use init method from TE
        with torch.no_grad():
            self.ln_weight.fill_(float(True))
            self.ln_bias.zero_()
    

    def forward(self, inputs):
        placeholder = inputs
        if "loss_scale_func" in GLOBAL_VARIABLES:
            loss_scale_func = GLOBAL_VARIABLES["loss_scale_func"]
        else:
            loss_scale_func = None
        placeholder = DSHB_.apply(placeholder, self.embedding, self.output_layer,self.ln_weight,self.ln_bias,self.logits_weight,self.loss_func,self.num_mb,loss_scale_func,self.stream_fwd,self.stream_bwd)
        return placeholder
