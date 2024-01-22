import torch
import torch.nn as nn
import torch.nn.functional as F


def log_mean(x, dim):
    return torch.logsumexp(x, dim=dim) - torch.log(
        torch.tensor(x.shape[dim], dtype=torch.float32)
    )


def entropy_reg(logits: torch.Tensor, mean_over_batch: bool = True):
    """Entropy regularization for the router."""

    entropy_l = lambda l: -(l * l.exp()).sum(-1)
    # softmax over experts
    # logits: [batch_size * sequence_length, num_experts]
    logprobs = F.log_softmax(logits, dim=-1)
    if mean_over_batch:
        # take mean probability over batch
        logprobs = log_mean(logprobs, 0)

    return -entropy_l(logprobs).mean()


# two losses below are adapted from
# https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/routing.py
def load_balancing_loss(logits: torch.Tensor, expert_indices: torch.Tensor) -> float:
    """Computes auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
    implements the loss function presented in equations (4) - (6). It aims to
    penalize those cases where the routing between experts is unbalanced.

    Args:
      logits: logits assigned to each expert per token. Shape:
        <float32>[batch_size * sequence_length, num_experts].
      expert_indices: <int>[batch_size * sequence_length, num_selected_experts]
        indices identifying the top num_selected_experts for a given token.

    Returns:
      The auxiliary loss.
    """
    # num_token = batch_size * sequence_length
    num_token, num_experts = logits.shape

    # Shape: [batch_size * sequence_length, num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_indices, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [batch_size * sequence_length, num_experts]
    expert_mask, _ = torch.max(expert_mask, dim=-2)

    # shape [num_experts]
    tokens_per_expert = torch.mean(expert_mask, dim=0, dtype=torch.float32)

    # compute router probability per expert in log space for numerical stability
    logprobs = F.log_softmax(logits, dim=-1)
    # take mean probability over batch
    # shape [num_experts]
    logprobs = log_mean(logprobs, dim=0)
    router_prob_per_expert = torch.exp(logprobs)
    return (
        torch.mean(  # mean over experts
            tokens_per_expert * router_prob_per_expert,
            dtype=torch.float32,
        )
        * num_experts
    )


def router_z_loss(router_logits: torch.Tensor) -> float:
    """Compute router z-loss.

     The router z-loss was introduced in Designing Effective Sparse Expert Models
     (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
     small in an effort to improve stability.

    Args:
      router_logits: <float>[batch_size * sequence_length, num_experts]
        router logits

    Returns:
      Scalar router z-loss.
    """
    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss, dtype=torch.float32) / (num_tokens)
