# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)

from .utils import VocabUtility



class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, lookup_probabilities, label_smoothing=0.0):
        '''
        Compute loss_u = exp(ppl_u)
        where ppl_u =\prod_{t=1}^{T} [p(w_t|w_1:t-1)/p(w_t)]^(-1/T)
        For comparison, ppl = \prod_{t=1}^{T} [p(w_t|w_1:t-1)]^(-1/T)
        '''

        assert lookup_probabilities is not None

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_tensor_model_parallel_group())
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index # [S,B]
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1) # [S*B]
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_tensor_model_parallel_group())

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits # [S,B,V/tp]
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_tensor_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.

        probabilities = lookup_probabilities[target] #[S,B]  
        loss = torch.log(sum_exp_logits * probabilities) - predicted_logits
        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0: # set as 0
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, lookup_probabilities, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Arguments:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        lobal_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, lookup_probabilities, label_smoothing)


def vocab_parallel_max_indices(logits):
    """
    Performs argmax(dim=-1) over logits across tensor parallel ranks
    Arguments:
        logits: logits split across tensor parallel ranks
                dimension is [sequence_length, batch_size, hidden_size]
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return logits.argmax(dim=-1)

    seq_length, batch_size, partition_vocab_size = logits.shape
    max_values, max_indices = logits.max(dim=-1)

    # Get the partition's vocab indices
    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    rank = get_tensor_model_parallel_rank()
    vocab_start_index, _ = get_vocab_range(partition_vocab_size, rank, world_size)
    max_indices = max_indices + vocab_start_index

    # gather max values and indices of all ranks
    max_values_group = torch.zeros(world_size, seq_length, batch_size, dtype=logits.dtype, device=logits.device)
    max_indices_group = torch.zeros(world_size, seq_length, batch_size, dtype=torch.int64, device=logits.device)
    torch.distributed.all_gather_into_tensor(max_values_group, max_values, group=get_tensor_model_parallel_group())
    torch.distributed.all_gather_into_tensor(max_indices_group, max_indices, group=get_tensor_model_parallel_group())

    # find rank with maximum value for each position and gather corresponding indices
    max_group_indices = torch.argmax(max_values_group, dim=0, keepdim=True)
    max_indices = torch.gather(max_indices_group, dim=0, index=max_group_indices).squeeze(0)
    return max_indices
