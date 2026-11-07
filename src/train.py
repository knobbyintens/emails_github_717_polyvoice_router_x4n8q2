"""Training and load balancing for PolyVoice."""

import yaml
import torch
from .model import PolyVoiceModel


def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    importance = weights.mean(dim=(0, 1))
    load = (weights > 0).float().mean(dim=(0, 1))
    return (importance * load).sum() * weights.size(-1)


def train(cfg):
    model = PolyVoiceModel(cfg["hidden_size"], cfg["num_experts"], cfg["num_languages"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for step in range(cfg["steps"]):
        features = torch.randn(2, 400, 80)
        language_ids = torch.randint(0, cfg["num_languages"], (2,))
        out = model(features, language_ids)
        loss = out.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        train(yaml.safe_load(handle))
