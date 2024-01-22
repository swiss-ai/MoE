"""
Simple MoE routing implementations that replace the MLP block in a standard transformer.
References:


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """
    Simplest MoE implementation with a linear router and softmax over experts.

    Note that in this implementation, we simply loop over the experts and
    aggregate the results. This is not the most efficient way to do it, but
    it also avoids the large memory overhead _and_ has no token dropping
    (because we do not need the capacity factor).
    """

    def __init__(self, config, mlp):
        super().__init__()
        assert config.moe_num_experts > 0
        self.experts = nn.ModuleList(
            [mlp(config=config) for _ in range(config.moe_num_experts)]
        )
        self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
        self.top_k = config.moe_num_experts_per_tok
        self.softmax_order = config.moe_softmax_order

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            weights, selected_experts = torch.topk(all_probs, self.top_k)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.top_k)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        results = torch.zeros_like(inputs_squashed)
        # naive looping over experts
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            output, _ = expert(inputs_squashed[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * output

        # return results and router logits (for aux loss calculation later)
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
        }


class ExpertChoiceMoE(nn.Module):
    """
    This is the MoE implementation that uses the expert choice method from
    https://arxiv.org/pdf/2202.09368v2.pdf.

    The main difference is that the router takes the softmax over the tokens, not the experts
    (i.e. each expert chooses its top-k tokens, not the other way around).
    For the same capacity factor, in theory, the same compute will be used as in standard top-k routing.
    AFAICT, there is no way around the capacity factor (whereas the code above does not need it).
    """

    def __init__(self, config, mlp):
        super().__init__()
        assert config.moe_num_experts > 0
        self.n_experts = config.moe_num_experts
        self.experts = nn.ModuleList(
            [mlp(config=config) for _ in range(config.moe_num_experts)]
        )
        self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
        self.capacity_factor = config.capacity_factor
        self.softmax_order = config.moe_softmax_order
        self.top_k = int(
            self.capacity_factor
            * config.batch_size
            * config.sequence_length
            / config.moe_num_experts
        )

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        num_tokens = inputs_squashed.shape[0]
        top_k = min(self.top_k, int(self.capacity_factor * num_tokens / self.n_experts))
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=0, dtype=torch.float32)
            # selection over tokens!
            # weights and selected tokens: [num_experts, top_k]
            weights, selected_tokens = torch.topk(all_probs.T, top_k)
        elif self.softmax_order == "topk_softmax":
            # weights and selected tokens: [num_experts, top_k]
            weights, selected_tokens = torch.topk(router_logits.T, top_k)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        """ this is the full parallel version with einsum -- this can OOM quickly """
        # [num_experts, top_k, num_tokens]
        # P = F.one_hot(selected_tokens, num_tokens).type_as(inputs_squashed)
        # # [num_experts, top_k, n_embd]
        # x_in = torch.matmul(P, inputs_squashed)
        # # [num_experts, num_tokens, n_embd]
        # experts_out = torch.stack(
        #     [expert(x)[0] for expert, x in zip(self.experts, x_in)], dim=0
        # )
        # results = torch.einsum("ijl,ij,ijd->ld", P, weights, experts_out)

        """ this is the naive loop version """
        # need to loop through experts because of memory growing too large
        # when doing everything in parallel?
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            # [top_k]
            batch_idx = selected_tokens[i]
            # [top_k, n_embd]
            output, _ = expert(inputs_squashed[batch_idx])
            results[batch_idx] += weights[i, :, None] * output

        # return results and router logits (for aux loss calculation later)
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_tokens,
        }
