# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from areal.utils.functional import fused_linear_logprobs_entropy


def _reference_logprobs_entropy(hidden, weight, labels, bias=None, temperature=1.0):
    logits = hidden @ weight.t()
    if bias is not None:
        logits = logits + bias
    logits = logits / temperature
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return logprobs, entropy


@pytest.mark.parametrize("use_bias", [False, True])
def test_fused_linear_logprobs_entropy_matches_reference(use_bias):
    torch.manual_seed(0)
    hidden = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(13, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 13, (2, 3))
    bias = (
        torch.randn(13, dtype=torch.float64, requires_grad=True) if use_bias else None
    )

    logprobs, entropy = fused_linear_logprobs_entropy(
        hidden,
        weight,
        labels,
        bias=bias,
        temperature=0.7,
        vocab_chunk_size=4,
        token_chunk_size=2,
    )
    expected_logprobs, expected_entropy = _reference_logprobs_entropy(
        hidden, weight, labels, bias=bias, temperature=0.7
    )

    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-10, atol=1e-10)


def test_fused_linear_logprobs_entropy_gradients_match_reference():
    torch.manual_seed(1)
    hidden = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(17, 6, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(17, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 17, (5,))
    grad_logprobs = torch.randn(5, dtype=torch.float64)
    grad_entropy = torch.randn(5, dtype=torch.float64)

    logprobs, entropy = fused_linear_logprobs_entropy(
        hidden,
        weight,
        labels,
        bias=bias,
        temperature=1.3,
        vocab_chunk_size=5,
        token_chunk_size=3,
    )
    loss = (logprobs * grad_logprobs + entropy * grad_entropy).sum()
    loss.backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()
    actual_bias_grad = bias.grad.detach().clone()

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)
    ref_logprobs, ref_entropy = _reference_logprobs_entropy(
        ref_hidden, ref_weight, labels, bias=ref_bias, temperature=1.3
    )
    ref_loss = (ref_logprobs * grad_logprobs + ref_entropy * grad_entropy).sum()
    ref_loss.backward()

    torch.testing.assert_close(
        actual_hidden_grad, ref_hidden.grad, rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(
        actual_weight_grad, ref_weight.grad, rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(actual_bias_grad, ref_bias.grad, rtol=1e-10, atol=1e-10)


def test_fused_linear_logprobs_entropy_handles_large_logits():
    torch.manual_seed(2)
    hidden = (torch.randn(4, 5) * 20).requires_grad_(True)
    weight = (torch.randn(19, 5) * 20).requires_grad_(True)
    labels = torch.randint(0, 19, (4,))

    logprobs, entropy = fused_linear_logprobs_entropy(
        hidden,
        weight,
        labels,
        vocab_chunk_size=3,
        token_chunk_size=2,
    )
    expected_logprobs, expected_entropy = _reference_logprobs_entropy(
        hidden, weight, labels
    )

    assert torch.isfinite(logprobs).all()
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-5)
