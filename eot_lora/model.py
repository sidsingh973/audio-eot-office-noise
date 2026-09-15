"""Whisper-tiny encoder (8 s window) + LoRA adapters + Smart Turn's pooling/classifier head.

The head mirrors smart-turn/train.py:73-87. Unlike Smart Turn, the pretrained encoder is
frozen and only LoRA adapters on its attention and MLP projections are trained.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import WhisperConfig, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

BASE = "openai/whisper-tiny"
N_POSITIONS = 400  # 8 s -> 800 log-mel frames -> 400 encoder frames (conv2 has stride 2)
LORA_TARGETS = r"encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|fc1|fc2)"


class EoTModel(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        config.max_source_positions = N_POSITIONS
        self.encoder = WhisperEncoder(config)
        d = config.d_model
        self.pool_attention = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )
        for m in [*self.pool_attention, *self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """input_features [B, 80, 800] log-mel -> logits [B] (sigmoid gives P(turn complete))."""
        h = self.encoder(input_features=input_features).last_hidden_state
        w = torch.softmax(self.pool_attention(h), dim=1)
        return self.classifier((h * w).sum(dim=1)).squeeze(-1)


def build(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    config = WhisperConfig.from_pretrained(BASE)
    model = EoTModel(config)
    state = WhisperModel.from_pretrained(BASE).encoder.state_dict()
    # Positions are fixed sinusoids, so the first 400 rows are exactly an 8 s table.
    state["embed_positions.weight"] = state["embed_positions.weight"][:N_POSITIONS]
    model.encoder.load_state_dict(state, strict=True)
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                      modules_to_save=["pool_attention", "classifier"])
    return get_peft_model(model, lora)


def save(model, path) -> None:
    """Save only what training changed: LoRA weights + the head (~2.6 MB)."""
    torch.save(get_peft_model_state_dict(model), path)


def load(path, device: str = "cpu"):
    model = build()
    set_peft_model_state_dict(model, torch.load(path, map_location="cpu"))
    return model.to(device).eval()
