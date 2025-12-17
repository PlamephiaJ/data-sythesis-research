import torch
import torch.nn as nn
from transformers import AutoModel


class CaptionEncoder(nn.Module):

    def __init__(
        self,
        name: str,
        cond_dim: int,
        pool: str = "cls",  # "cls" or "mean"
        dropout: float = 0.0,
        freeze: bool = True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(name)
        self.pool = pool
        self.dropout = nn.Dropout(dropout)

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, cond_dim)

        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

    @torch.no_grad()
    def _encode_frozen(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def forward(self, input_ids=None, attention_mask=None):
        """
        Returns:
            cond: Tensor[B, cond_dim]
        """
        if input_ids is None:
            return self.null_cond.unsqueeze(0)

        if not self.encoder.training and not any(
            p.requires_grad for p in self.encoder.parameters()
        ):
            hidden = self._encode_frozen(input_ids, attention_mask)
        else:
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        if self.pool == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).type_as(hidden)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.proj(self.dropout(pooled))
