"""Language-aware mixture-of-experts router."""

import torch
import torch.nn as nn


class LanguageRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_languages: int):
        super().__init__()
        self.language_embedding = nn.Embedding(num_languages, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, num_experts, bias=False)

    def forward(self, states: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        lang = self.language_embedding(language_ids)
        lang = lang.unsqueeze(1).expand(-1, states.size(1), -1)
        logits = self.gate(torch.cat([states, lang], dim=-1))
        return torch.softmax(logits, dim=-1)
