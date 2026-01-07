import torch
import torch.nn as nn
from transformers import AutoModel


def l2_normalize(x, eps=1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


class CaptionEncoder(nn.Module):
    """
    Supports:
      - pooled conditioning: forward(...) -> Tensor[B, cond_dim]
      - pooled + token embeddings for cross-attn:
            forward(..., return_tokens=True, token_dim=<model_hidden>) ->
                (pooled_cond: Tensor[B, cond_dim],
                 token_emb:   Tensor[B, token_dim],
                 token_mask:  Tensor[B, L] (same as attention_mask, returned for convenience))

    Notes:
      - token_dim defaults to the transformer's hidden size; you can set token_dim to your DDiT dim
        (e.g., config.model.hidden_size) so cross-attn KV matches block dim.
      - In freeze=True mode, _encode_frozen runs under no_grad, same behavior as before.
      - If input_ids is None:
          pooled returns null_cond.unsqueeze(0) (same as your previous behavior).
          return_tokens=True returns a single "null token" embedding and a mask of length 1.
    """

    def __init__(
        self,
        name: str,
        cond_dim: int,
        pool: str = "cls",  # "cls" or "mean"
        dropout: float = 0.0,
        freeze: bool = True,
        token_dim: (
            int | None
        ) = None,  # set to DDiT hidden_size for cross-attn; default = encoder hidden
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(name)
        self.pool = pool
        self.dropout = nn.Dropout(dropout)

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        enc_hidden = self.encoder.config.hidden_size

        # pooled projection (to cond_dim)
        self.proj = nn.Linear(enc_hidden, cond_dim)

        # token projection (to token_dim)
        if token_dim is None:
            token_dim = enc_hidden
        self.token_dim = int(token_dim)
        self.token_proj = (
            nn.Identity()
            if self.token_dim == enc_hidden
            else nn.Linear(enc_hidden, self.token_dim, bias=False)
        )

        # null conditioning (pooled)
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        # null token (token-level) for cross-attn when caption is missing
        self.null_token = nn.Parameter(torch.zeros(self.token_dim))

    @torch.no_grad()
    def _encode_frozen(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *,
        return_tokens: bool = False,
        return_align: bool = False,
    ):
        """
        Args:
            input_ids: LongTensor[B, L] or None
            attention_mask: Long/BoolTensor[B, L] (1 valid, 0 pad) or None
            return_tokens: if True, also return token embeddings for cross-attn

        Returns:
            if return_tokens=False:
                pooled_cond: Tensor[B, cond_dim]
            if return_tokens=True:
                (pooled_cond: Tensor[B, cond_dim],
                 token_emb:   Tensor[B, L, token_dim],
                 token_mask:  Tensor[B, L])
        """
        if input_ids is None:
            pooled = self.null_cond.unsqueeze(0)  # (1, cond_dim)
            if not return_tokens:
                if return_align:
                    pooled = l2_normalize(pooled)
                return pooled
            # Provide a single "null token" to keep shapes sane for cross-attn
            token_emb = self.null_token.view(1, 1, self.token_dim)  # (1,1,token_dim)
            token_mask = torch.ones((1, 1), device=token_emb.device, dtype=torch.int64)
            if return_align:
                pooled = l2_normalize(pooled)
            return pooled, token_emb, token_mask

        if attention_mask is None:
            # assume all valid if not provided
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=input_ids.device
            )

        # encode (frozen or trainable)
        if (not self.encoder.training) and (
            not any(p.requires_grad for p in self.encoder.parameters())
        ):
            hidden = self._encode_frozen(input_ids, attention_mask)  # (B,L,enc_hidden)
        else:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state  # (B,L,enc_hidden)

        # pooled
        if self.pool == "cls":
            pooled_raw = hidden[:, 0]  # (B, enc_hidden)
        else:
            mask = attention_mask.unsqueeze(-1).type_as(hidden)
            pooled_raw = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        pooled_cond = self.proj(self.dropout(pooled_raw))  # (B, cond_dim)

        if return_align:
            pooled_cond = l2_normalize(pooled_cond)

        if not return_tokens:
            return pooled_cond

        # tokens for cross-attn KV
        token_hidden = self.dropout(hidden)  # (B,L,enc_hidden)
        token_emb = self.token_proj(token_hidden)  # (B,L,token_dim)
        token_mask = attention_mask  # (B,L)
        return pooled_cond, token_emb, token_mask
