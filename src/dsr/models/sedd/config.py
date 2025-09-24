"""Configuration objects for SEDD models."""

from typing import Any

from transformers import PretrainedConfig


class SEDDConfig(PretrainedConfig):
    model_type = "sedd"

    def __init__(
        self,
        tokens: int,
        graph_type: str = "absorb",
        hidden_size: int = 768,
        cond_dim: int = 128,
        length: int = 1024,
        n_blocks: int = 12,
        n_heads: int = 12,
        scale_by_sigma: bool = True,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tokens = tokens
        self.graph_type = graph_type
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.length = length
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.scale_by_sigma = scale_by_sigma
        self.dropout = dropout
        self.vocab_size = tokens + (1 if graph_type == "absorb" else 0)
