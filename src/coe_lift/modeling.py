"""Model-side modules for CoE-LIFT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoEFeatures:
    pooled: torch.Tensor
    layer_vectors: torch.Tensor
    scalar_features: torch.Tensor


class CoEProjector(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * n_layers, 1024),
            nn.SiLU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class OODGate(nn.Module):
    def __init__(self, feature_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class GatedSoftLatentPrompt(nn.Module):
    """Learned soft latent tokens scaled by an OOD probability.

    The tokens are prepended to every sequence. A low gate probability drives their
    embeddings near zero; a high probability lets the prompt allocate additional
    continuous reasoning capacity.
    """

    def __init__(self, num_tokens: int, hidden_size: int, init_std: float = 0.02):
        super().__init__()
        self.num_tokens = num_tokens
        self.latents = nn.Parameter(torch.empty(num_tokens, hidden_size))
        nn.init.normal_(self.latents, mean=0.0, std=init_std)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        gate_prob: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.num_tokens <= 0:
            return token_embeddings, attention_mask, labels

        batch_size = token_embeddings.size(0)
        if gate_prob is None:
            gate_prob = token_embeddings.new_ones(batch_size)
        latent = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latent = latent * gate_prob.view(batch_size, 1, 1).to(latent.dtype)
        inputs_embeds = torch.cat([latent, token_embeddings], dim=1)

        latent_mask = attention_mask.new_ones(batch_size, self.num_tokens)
        attention_mask = torch.cat([latent_mask, attention_mask], dim=1)

        if labels is not None:
            latent_labels = labels.new_full((batch_size, self.num_tokens), -100)
            labels = torch.cat([latent_labels, labels], dim=1)
        return inputs_embeds, attention_mask, labels


def parse_layers(layer_arg: str, n_hidden_states: int | None = None) -> list[int]:
    layer_arg = layer_arg.strip()
    if layer_arg.lower() == "all":
        if n_hidden_states is None:
            raise ValueError('Layer argument "all" requires n_hidden_states.')
        return list(range(1, n_hidden_states))

    layers = [int(part.strip()) for part in layer_arg.split(",") if part.strip()]
    if n_hidden_states is None:
        return layers

    max_idx = n_hidden_states - 1
    resolved = [idx if idx >= 0 else max_idx + 1 + idx for idx in layers]
    invalid = [idx for idx in resolved if idx < 0 or idx > max_idx]
    if invalid:
        raise ValueError(
            f"Layer indices {invalid} are outside hidden-state range [0, {max_idx}]. "
            "Use relative indices such as -18,-12,-6,-1 for model-agnostic upper layers."
        )
    return resolved


def extract_coe_features(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layers: list[int],
) -> CoEFeatures:
    idx = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
    layer_vectors = torch.stack(
        [hidden_states[layer][batch_indices, idx] for layer in layers],
        dim=1,
    )
    pooled = layer_vectors.flatten(start_dim=1)
    scalar_features = coe_scalar_features(layer_vectors)
    return CoEFeatures(pooled=pooled, layer_vectors=layer_vectors, scalar_features=scalar_features)


def coe_scalar_features(layer_vectors: torch.Tensor) -> torch.Tensor:
    abs_values = layer_vectors.abs()
    sparsity = (abs_values < abs_values.mean(dim=(1, 2), keepdim=True)).float().mean(dim=(1, 2))

    if layer_vectors.size(1) >= 3:
        normed = F.normalize(layer_vectors, dim=-1)
        curvature = (normed[:, 2:] - 2 * normed[:, 1:-1] + normed[:, :-2]).norm(dim=-1).mean(dim=1)
    else:
        curvature = layer_vectors.new_zeros(layer_vectors.size(0))

    final = F.normalize(layer_vectors[:, -1], dim=-1)
    centroid = final.mean(dim=0, keepdim=True)
    drift = 1.0 - F.cosine_similarity(final, centroid, dim=-1)
    return torch.stack([sparsity, curvature, drift], dim=-1)


def group_info_nce(z: torch.Tensor, group_ids: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.new_zeros(())

    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(z.size(0), device=z.device, dtype=torch.bool), -1e9)
    same = group_ids[:, None].eq(group_ids[None, :])
    same = same.masked_fill(torch.eye(z.size(0), device=z.device, dtype=torch.bool), False)
    positive_counts = same.sum(dim=1)
    if not torch.any(positive_counts > 0):
        return z.new_zeros(())

    logp = logits - logits.logsumexp(dim=1, keepdim=True)
    per_row = -(logp * same).sum(dim=1) / positive_counts.clamp_min(1)
    return per_row[positive_counts > 0].mean()


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    if torch.distributed.get_world_size() == 1:
        return tensor
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(tensor)
    return torch.cat(tuple(gathered), dim=0)


def gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    if torch.distributed.get_world_size() == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def effective_rank_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if min(z.shape) <= 1:
        return z.new_zeros(())
    centered = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    probs = singular_values / singular_values.sum().clamp_min(eps)
    entropy = -(probs * (probs + eps).log()).sum()
    max_entropy = torch.log(torch.tensor(float(min(z.shape)), device=z.device))
    return (max_entropy - entropy).to(z.dtype)
