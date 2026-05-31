# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import distributed as dist

from areal.infra.platforms import is_npu_available

T = TypeVar("T", torch.Tensor, tuple[torch.Tensor, torch.Tensor])


def _gather_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


def _gather_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


def _should_use_torch_compile() -> bool:
    return not is_npu_available


if _should_use_torch_compile():
    _gather_logprobs = torch.compile(_gather_logprobs)
    _gather_logprobs_entropy = torch.compile(_gather_logprobs_entropy)


def _chunked_apply(
    fn: Callable[[torch.Tensor, torch.Tensor], T],
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> T:
    """Apply a function in chunks along the first dimension to reduce peak memory."""
    total_seqlen = logits.shape[0]
    assert total_seqlen > 0, "Input logits must have at least one element"
    results: list = []

    for i in range(0, total_seqlen, chunk_size):
        end_idx = min(i + chunk_size, total_seqlen)
        chunk_result = fn(logits[i:end_idx], labels[i:end_idx])
        results.append(chunk_result)

    # Handle single tensor vs tuple of tensors
    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        return tuple(torch.cat([r[i] for r in results]) for i in range(num_outputs))
    return torch.cat(results)


def _chunked_gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    fn = functools.partial(_gather_logprobs, temperature=temperature)
    return _chunked_apply(fn, logits, labels, chunk_size)


def _chunked_gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = functools.partial(_gather_logprobs_entropy, temperature=temperature)
    return _chunked_apply(fn, logits, labels, chunk_size)


class _FusedLinearLogProbsEntropy(torch.autograd.Function):
    """Memory-efficient final-linear log-probability and entropy computation.

    This follows the same high-level strategy as verl/Liger fused linear cross
    entropy kernels, but keeps the implementation in PyTorch for portability:
    stream the vocabulary projection in chunks, keep only row-wise normalization
    statistics, and recompute chunks during backward instead of saving logits.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor | None,
        temperature: float,
        tp_group: dist.ProcessGroup | None,
        vocab_chunk_size: int,
        token_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[:-1] != labels.shape:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} must match hidden leading shape "
                f"{tuple(hidden.shape[:-1])}"
            )
        if hidden.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"hidden size {hidden.shape[-1]} must match weight hidden size "
                f"{weight.shape[-1]}"
            )
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError(
                f"bias shape {tuple(bias.shape)} must match local vocab size "
                f"{weight.shape[0]}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if vocab_chunk_size <= 0 or token_chunk_size <= 0:
            raise ValueError(
                "vocab_chunk_size and token_chunk_size must both be positive"
            )

        original_shape = labels.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1])
        labels_1d = labels.reshape(-1)
        n_rows = hidden_2d.shape[0]
        local_vocab_size = weight.shape[0]
        compute_dtype = (
            torch.float32
            if hidden.dtype in (torch.float16, torch.bfloat16)
            else hidden.dtype
        )
        inv_temperature = 1.0 / float(temperature)

        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            tp_rank = dist.get_rank(tp_group)
            vocab_start = tp_rank * local_vocab_size
        else:
            vocab_start = 0

        logprobs = torch.empty(n_rows, device=hidden.device, dtype=compute_dtype)
        entropy = torch.empty(n_rows, device=hidden.device, dtype=compute_dtype)
        log_z = torch.empty(n_rows, device=hidden.device, dtype=compute_dtype)
        mean_logits = torch.empty(n_rows, device=hidden.device, dtype=compute_dtype)

        with torch.no_grad():
            for token_start in range(0, n_rows, token_chunk_size):
                token_end = min(token_start + token_chunk_size, n_rows)
                hidden_chunk = hidden_2d[token_start:token_end]
                labels_chunk = labels_1d[token_start:token_end]
                chunk_rows = token_end - token_start

                row_max = torch.full(
                    (chunk_rows,),
                    -float("inf"),
                    device=hidden.device,
                    dtype=compute_dtype,
                )
                sum_exp = torch.zeros(
                    chunk_rows, device=hidden.device, dtype=compute_dtype
                )
                sum_exp_logits = torch.zeros_like(sum_exp)
                selected_logits = torch.zeros_like(sum_exp)
                row_idx = torch.arange(chunk_rows, device=hidden.device)

                for vocab_start_local in range(0, local_vocab_size, vocab_chunk_size):
                    vocab_end_local = min(
                        vocab_start_local + vocab_chunk_size, local_vocab_size
                    )
                    weight_chunk = weight[vocab_start_local:vocab_end_local]
                    logits_chunk = (
                        hidden_chunk.to(compute_dtype)
                        @ weight_chunk.to(compute_dtype).t()
                    )
                    if bias is not None:
                        logits_chunk = logits_chunk + bias[
                            vocab_start_local:vocab_end_local
                        ].to(compute_dtype)
                    logits_chunk = logits_chunk * inv_temperature

                    chunk_max = logits_chunk.amax(dim=-1)
                    new_max = torch.maximum(row_max, chunk_max)
                    old_scale = torch.exp(row_max - new_max)
                    exp_logits = torch.exp(logits_chunk - new_max.unsqueeze(-1))
                    sum_exp = sum_exp * old_scale + exp_logits.sum(dim=-1)
                    sum_exp_logits = sum_exp_logits * old_scale + (
                        exp_logits * logits_chunk
                    ).sum(dim=-1)
                    row_max = new_max

                    global_start = vocab_start + vocab_start_local
                    global_end = vocab_start + vocab_end_local
                    in_chunk = (labels_chunk >= global_start) & (
                        labels_chunk < global_end
                    )
                    local_idx = torch.clamp(
                        labels_chunk - global_start,
                        min=0,
                        max=vocab_end_local - vocab_start_local - 1,
                    )
                    selected_logits = selected_logits + (
                        logits_chunk[row_idx, local_idx] * in_chunk.to(compute_dtype)
                    )

                if tp_group is not None and dist.get_world_size(tp_group) > 1:
                    global_max = row_max.clone()
                    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
                    rescale = torch.exp(row_max - global_max)
                    sum_exp = sum_exp * rescale
                    sum_exp_logits = sum_exp_logits * rescale
                    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
                    dist.all_reduce(
                        sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group
                    )
                    dist.all_reduce(
                        selected_logits, op=dist.ReduceOp.SUM, group=tp_group
                    )
                    row_max = global_max

                log_z_chunk = row_max + torch.log(sum_exp)
                mean_logits_chunk = sum_exp_logits / sum_exp
                log_z[token_start:token_end] = log_z_chunk
                mean_logits[token_start:token_end] = mean_logits_chunk
                logprobs[token_start:token_end] = selected_logits - log_z_chunk
                entropy[token_start:token_end] = log_z_chunk - mean_logits_chunk

        bias_to_save = bias if bias is not None else hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, labels, bias_to_save, log_z, mean_logits)
        ctx.has_bias = bias is not None
        ctx.temperature = float(temperature)
        ctx.tp_group = tp_group
        ctx.vocab_chunk_size = int(vocab_chunk_size)
        ctx.token_chunk_size = int(token_chunk_size)
        ctx.original_shape = original_shape

        return logprobs.reshape(original_shape), entropy.reshape(original_shape)

    @staticmethod
    def backward(
        ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        hidden, weight, labels, bias, log_z, mean_logits = ctx.saved_tensors
        hidden_2d = hidden.reshape(-1, hidden.shape[-1])
        labels_1d = labels.reshape(-1)
        compute_dtype = (
            torch.float32
            if hidden.dtype in (torch.float16, torch.bfloat16)
            else hidden.dtype
        )
        grad_logprobs_1d = grad_logprobs.reshape(-1).to(compute_dtype)
        grad_entropy_1d = grad_entropy.reshape(-1).to(compute_dtype)
        log_z = log_z.to(compute_dtype)
        mean_logits = mean_logits.to(compute_dtype)

        n_rows = hidden_2d.shape[0]
        local_vocab_size = weight.shape[0]
        inv_temperature = 1.0 / ctx.temperature
        has_bias = ctx.has_bias
        bias_tensor = bias if has_bias else None

        if ctx.tp_group is not None and dist.get_world_size(ctx.tp_group) > 1:
            tp_rank = dist.get_rank(ctx.tp_group)
            vocab_start = tp_rank * local_vocab_size
        else:
            vocab_start = 0

        grad_hidden = (
            torch.zeros_like(hidden_2d, dtype=compute_dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = (
            torch.zeros_like(weight, dtype=compute_dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        grad_bias = (
            torch.zeros(local_vocab_size, device=weight.device, dtype=compute_dtype)
            if has_bias and ctx.needs_input_grad[3]
            else None
        )

        for token_start in range(0, n_rows, ctx.token_chunk_size):
            token_end = min(token_start + ctx.token_chunk_size, n_rows)
            hidden_chunk = hidden_2d[token_start:token_end]
            labels_chunk = labels_1d[token_start:token_end]
            grad_logprobs_chunk = grad_logprobs_1d[token_start:token_end]
            grad_entropy_chunk = grad_entropy_1d[token_start:token_end]
            log_z_chunk = log_z[token_start:token_end]
            mean_logits_chunk = mean_logits[token_start:token_end]
            chunk_rows = token_end - token_start
            row_idx = torch.arange(chunk_rows, device=hidden.device)

            for vocab_start_local in range(0, local_vocab_size, ctx.vocab_chunk_size):
                vocab_end_local = min(
                    vocab_start_local + ctx.vocab_chunk_size, local_vocab_size
                )
                weight_chunk = weight[vocab_start_local:vocab_end_local]
                logits_chunk = (
                    hidden_chunk.to(compute_dtype) @ weight_chunk.to(compute_dtype).t()
                )
                if bias_tensor is not None:
                    logits_chunk = logits_chunk + bias_tensor[
                        vocab_start_local:vocab_end_local
                    ].to(compute_dtype)
                logits_chunk = logits_chunk * inv_temperature
                probs = torch.exp(logits_chunk - log_z_chunk.unsqueeze(-1))

                grad_logits = -grad_logprobs_chunk.unsqueeze(-1) * probs
                grad_logits = grad_logits + grad_entropy_chunk.unsqueeze(-1) * probs * (
                    mean_logits_chunk.unsqueeze(-1) - logits_chunk
                )

                global_start = vocab_start + vocab_start_local
                global_end = vocab_start + vocab_end_local
                in_chunk = (labels_chunk >= global_start) & (labels_chunk < global_end)
                local_idx = torch.clamp(
                    labels_chunk - global_start,
                    min=0,
                    max=vocab_end_local - vocab_start_local - 1,
                )
                grad_logits[row_idx, local_idx] += grad_logprobs_chunk * in_chunk.to(
                    grad_logits.dtype
                )

                grad_logits = grad_logits * inv_temperature

                if grad_hidden is not None:
                    grad_hidden[token_start:token_end].add_(
                        grad_logits @ weight_chunk.to(compute_dtype)
                    )
                if grad_weight is not None:
                    grad_weight[vocab_start_local:vocab_end_local].add_(
                        grad_logits.t() @ hidden_chunk.to(compute_dtype)
                    )
                if grad_bias is not None:
                    grad_bias[vocab_start_local:vocab_end_local].add_(
                        grad_logits.sum(dim=0)
                    )

        if grad_hidden is not None and ctx.tp_group is not None:
            if dist.get_world_size(ctx.tp_group) > 1:
                dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)

        return (
            grad_hidden.reshape_as(hidden).to(hidden.dtype)
            if grad_hidden is not None
            else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None,
            None,
            None,
            None,
        )


def fused_linear_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    bias: torch.Tensor | None = None,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute selected-token logprobs and entropy without materializing logits.

    Args:
        hidden: Final hidden states with shape ``[..., hidden_size]``.
        weight: Local lm-head weight with shape ``[vocab_size, hidden_size]`` or
            ``[vocab_size / tp_size, hidden_size]`` when ``tp_group`` is set.
        labels: Global token ids with shape matching ``hidden.shape[:-1]``.
        bias: Optional local lm-head bias.
        temperature: Positive softmax temperature.
        tp_group: Optional tensor-parallel group over vocabulary shards.
        vocab_chunk_size: Maximum local vocabulary columns projected at once.
        token_chunk_size: Maximum flattened tokens projected at once.

    Returns:
        A tuple of ``(logprobs, entropy)`` with shape ``labels.shape``.

    The function trades extra recomputation in backward for lower activation
    memory: peak logits memory is bounded by
    ``token_chunk_size * vocab_chunk_size`` instead of ``num_tokens * vocab``.
    """
    return _FusedLinearLogProbsEntropy.apply(
        hidden,
        weight,
        labels,
        bias,
        float(temperature),
        tp_group,
        int(vocab_chunk_size),
        int(token_chunk_size),
    )


class _VocabParallelLogProbs(torch.autograd.Function):
    """Compute log probabilities when logits are sharded on the vocab dimension.

    Given sharded logits [..., vocab_size/tp] and labels [...], computes:
        logprobs[i] = logits[i, labels[i]] - log(sum(exp(logits[i, :])))

    The input can have arbitrary leading dimensions (e.g., [batch, seq_len] or just
    [seq_len]). The labels indices are global (0 to vocab_size-1), and each TP rank
    only holds a partition of the vocabulary.

    Memory Optimization:
        Following Megatron's cross_entropy pattern, we use in-place operations to
        minimize memory allocations. The key optimization is in backward():

        - The gradient formula is: grad = one_hot(labels) - softmax
        - Since this only requires subtracting 1 at the label position and scaling,
          we can directly reuse the saved softmax tensor as grad_input (in-place).
        - This avoids allocating a new [*, vocab/tp] tensor for gradients.

        Forward saves only ONE large tensor:
        - softmax: [*, vocab/tp] - unavoidable for gradient computation

        Backward allocates NO new large tensors:
        - Reuses softmax directly as grad_input via in-place modifications

    Note:
        This implementation uses in-place operations on saved tensors for memory
        efficiency. As a result, it does NOT support:
        - `retain_graph=True` in backward()
        - Higher-order gradients (e.g., torch.autograd.grad with create_graph=True)

        These limitations are acceptable for typical RL training where only
        first-order gradients are needed and each backward is called once.
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # Get TP rank info
        tp_rank = dist.get_rank(tp_group)

        # Calculate vocab partition boundaries for this rank
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Step 1: Numerical stability - subtract max
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # In-place subtraction following Megatron pattern
        normalized_logits = vocab_parallel_logits - logits_max

        # Step 2: Compute exp in-place and sum across all ranks
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 3: Get the logit value at labels position
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 4: Compute log probability
        log_sum_exp = sum_exp_logits.squeeze(-1).log()
        logprobs = predicted_logits - log_sum_exp

        # Step 5: Compute softmax in-place for backward (reuse exp_logits memory)
        softmax = exp_logits.div_(sum_exp_logits)
        ctx.save_for_backward(softmax, labels_mask, masked_labels_1d)

        return logprobs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        softmax, labels_mask, masked_labels_1d = ctx.saved_tensors

        # Gradient of logprobs w.r.t. logits: one_hot(labels) - softmax
        # Following Megatron's pattern: use softmax directly as grad_input base
        # and modify in-place where possible
        partition_vocab_size = softmax.size(-1)

        # Use softmax as the gradient base (will be modified)
        grad_input = softmax
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        # Subtract 1 at labels position (only for labels in this partition)
        # This gives: softmax - one_hot(labels)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] -= update_mask.float()

        # Scale by grad_output (in-place)
        # Note: we want -(softmax - one_hot) = one_hot - softmax for logprobs gradient
        grad_input.mul_(grad_output.unsqueeze(-1))
        grad_input.neg_()

        return grad_input, None, None


class _VocabParallelLogProbsEntropy(torch.autograd.Function):
    """Compute both log probabilities and entropy when logits are sharded.

    Input tensors can have arbitrary leading dimensions:
        - logits: [..., vocab_size/tp]
        - labels: [...]

    This combines the computation to share intermediate results (softmax, sum_exp, etc.)
    and reduce redundant all-reduce operations compared to calling logprobs and entropy
    separately.

    Memory Optimization:
        Forward saves only ONE large tensor (softmax) plus a few small scalars.
        The entropy gradient is algebraically rewritten to avoid saving original logits:

            grad_entropy = softmax * (E[x] - x)
                         = softmax * (E[x] - log(softmax) - log(Z))
                         = softmax * (E[x] - log(Z)) - softmax * log(softmax)

        where E[x] = sum(softmax * logits) and log(Z) = log(sum(exp(logits))).

        Why we CANNOT reuse softmax in-place (unlike _VocabParallelLogProbs):
            The combined gradient requires multiple reads of the original softmax:

            1. grad_input = softmax * (E[x] - log(Z))   # first read
            2. grad_input -= xlogy(softmax, softmax)    # second read
            3. grad_input -= softmax * grad_logprobs    # third read

            If we modified softmax in step 1, steps 2 and 3 would get wrong values.
            In contrast, _VocabParallelLogProbs only needs: grad = softmax - one_hot,
            which can be done by subtracting 1 at one position then scaling - a single
            pass that allows full in-place reuse.

        Backward allocates ONE new large tensor:
            - grad_input: [*, vocab/tp] - created via `softmax * mean_x_minus_log_z`

        Memory comparison (seq=8192, vocab=152K, tp=2, fp32):
            - Naive approach: save both logits and softmax = ~4.7GB
            - Our approach: save only softmax = ~2.3GB (50% reduction in forward)
            - Backward: +2.3GB temporary for grad_input (unavoidable for correctness)

    Note:
        This implementation does NOT support:
        - `retain_graph=True` in backward()
        - Higher-order gradients (e.g., torch.autograd.grad with create_graph=True)

        These limitations are acceptable for typical RL training where only
        first-order gradients are needed and each backward is called once.

    Returns:
        logprobs: [...] log probability of labels tokens
        entropy: [...] entropy of the distribution
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get TP rank info
        tp_rank = dist.get_rank(tp_group)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Step 1: Numerical stability - subtract max (shared)
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # In-place subtraction following Megatron pattern
        normalized_logits = vocab_parallel_logits - logits_max

        # Step 2: Compute exp and sum_exp (shared)
        # Use in-place exp to reuse memory
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 3: Compute softmax in-place (shared)
        # After this, exp_logits becomes softmax
        softmax = exp_logits.div_(sum_exp_logits)

        # Step 4: For logprobs - get labels logit
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 5: For entropy - compute sum(softmax * logits)
        # Note: vocab_parallel_logits is the original (un-normalized) logits
        sum_softmax_times_logits = (softmax * vocab_parallel_logits).sum(
            dim=-1, keepdim=True
        )
        dist.all_reduce(sum_softmax_times_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 6: Compute final results
        log_sum_exp = sum_exp_logits.log()
        logprobs = predicted_logits - log_sum_exp.squeeze(-1)
        # entropy = log(Z) - E[x] = (max + log(sum_exp)) - sum_softmax_times_logits
        entropy = (logits_max + log_sum_exp - sum_softmax_times_logits).squeeze(-1)

        # Compute log(Z) for backward (small tensor: [*, 1])
        # log(Z) = max + log(sum_exp)
        log_z = logits_max + log_sum_exp

        # Save for backward - only ONE large tensor (softmax) instead of two
        # Memory savings: ~2.3GB for typical configs (seq=8192, vocab=152K, tp=2)
        ctx.save_for_backward(
            softmax,  # [*, vocab/tp] - the only large tensor
            sum_softmax_times_logits,  # [*, 1] - small
            log_z,  # [*, 1] - small
            labels_mask,  # [*] - small (bool)
            masked_labels_1d,  # [N] - small (int64)
        )
        ctx.partition_vocab_size = partition_vocab_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor) -> tuple:
        (
            softmax,
            sum_softmax_times_logits,
            log_z,
            labels_mask,
            masked_labels_1d,
        ) = ctx.saved_tensors
        partition_vocab_size = ctx.partition_vocab_size

        # Memory-optimized backward using in-place operations.
        # We compute gradients directly on softmax tensor to avoid extra allocations.
        #
        # Total gradient = grad_logprobs * (one_hot - softmax) + grad_entropy * softmax * (E[x] - x)
        #
        # Strategy: First compute entropy gradient (needs original softmax values),
        # then add logprobs gradient.

        # Step 1: Compute entropy gradient contribution
        # grad_entropy_contrib = softmax * ((E[x] - log(Z)) - log(softmax))
        #                     = softmax * (mean_x - log_z) - softmax * log(softmax)
        # Note: torch.xlogy handles 0 * log(0) = 0 correctly
        mean_x_minus_log_z = sum_softmax_times_logits - log_z  # [*, 1] small tensor

        # The gradient is computed in a single large tensor, grad_input, to minimize
        # peak memory usage. It is initialized here and then modified in-place.
        # Compute: softmax * (mean_x - log_z) - xlogy(softmax, softmax)
        # First: grad_input = softmax * mean_x_minus_log_z (broadcast, creates new tensor)
        grad_input = softmax * mean_x_minus_log_z
        # Subtract xlogy term in-place
        grad_input.sub_(torch.xlogy(softmax, softmax))
        # Scale by grad_entropy in-place
        grad_input.mul_(grad_entropy.unsqueeze(-1))

        # Step 2: Add logprobs gradient contribution
        # grad_logprobs_contrib = grad_logprobs * (one_hot(labels) - softmax)
        #                      = -grad_logprobs * softmax + grad_logprobs * one_hot
        # Add -softmax * grad_logprobs term
        grad_input.sub_(softmax * grad_logprobs.unsqueeze(-1))

        # Add one_hot * grad_logprobs at labels positions (only for labels in this partition)
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] += update_mask * grad_logprobs.view(-1)

        return grad_input, None, None


def _vocab_parallel_logprobs(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature != 1.0:
        logits = vocab_parallel_logits.float() / temperature
    else:
        logits = vocab_parallel_logits.float()
    return _VocabParallelLogProbs.apply(logits, labels, tp_group)


def _vocab_parallel_logprobs_entropy(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature != 1.0:
        logits = vocab_parallel_logits.float() / temperature
    else:
        logits = vocab_parallel_logits.float()
    return _VocabParallelLogProbsEntropy.apply(logits, labels, tp_group)


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute log probabilities with optional vocab parallelism for FSDP.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices with shape [...] for which to compute log probabilities.
        temperature: Softmax temperature scaling. Default is 1.0.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation
            to avoid gathering the full vocab dimension across TP ranks.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.

    Returns:
        Log probabilities at the label positions with shape [...].
    """
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(
            _vocab_parallel_logprobs,
            tp_group=tp_group,
            temperature=temperature,
        )
        return _chunked_apply(fn, logits, labels, chunk_size)

    return _chunked_gather_logprobs(logits, labels, temperature, chunk_size)


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy with optional vocab parallelism for FSDP.

    This function computes both values in a single pass, sharing intermediate results
    (softmax, sum_exp, etc.) to reduce redundant computation and all-reduce operations.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices with shape [...] for which to compute log probabilities.
        temperature: Softmax temperature scaling. Default is 1.0.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation
            to avoid gathering the full vocab dimension across TP ranks.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.

    Returns:
        A tuple of (logprobs, entropy):
            - logprobs: Log probabilities at the label positions with shape [...].
            - entropy: Entropy of the probability distribution with shape [...].
    """
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(
            _vocab_parallel_logprobs_entropy,
            tp_group=tp_group,
            temperature=temperature,
        )
        return _chunked_apply(fn, logits, labels, chunk_size)

    return _chunked_gather_logprobs_entropy(logits, labels, temperature, chunk_size)
