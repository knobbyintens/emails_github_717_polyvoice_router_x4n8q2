"""Speech translation model with a PolyVoice routing layer."""

import torch
import torch.nn as nn
from .router import LanguageRouter


class PolyVoiceModel(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_languages: int):
        super().__init__()
        self.encoder = nn.Linear(80, hidden_size)
        self.router = LanguageRouter(hidden_size, num_experts, num_languages)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]
        )
        self.decoder = nn.Linear(hidden_size, 1000)

    def forward(self, features: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        states = self.encoder(features)
        weights = self.router(states, language_ids)
        expert_out = torch.stack(
            [expert(states) for expert in self.experts], dim=-1
        )
        mixed = (expert_out * weights.unsqueeze(-2)).sum(dim=-1)
        return self.decoder(mixed)
