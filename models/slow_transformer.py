import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Learned attention pooling over semantic query tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, D], got {tuple(x.shape)}")
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.einsum("bt,btd->bd", weights, x)


class ClassAttentionResidualBlock(nn.Module):
    """Pre-norm semantic query attention over selected keyframe tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_residual_gate: bool = True,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.class_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = (
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            if use_residual_gate
            else None
        )
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        ff_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query)
        kv = self.token_norm(tokens)
        ctx, _ = self.class_attn(q, kv, kv, need_weights=False)
        if self.gate is not None:
            gate = self.gate(torch.cat([query, ctx], dim=-1))
            query = query + self.drop(gate * ctx)
        else:
            query = query + self.drop(ctx)
        return query + self.ffn(self.ffn_norm(query))


class SelectedClassAttentionEncoder(nn.Module):
    """Selected-only slow semantic encoder with class-attention queries.

    Selected keyframe tokens first interact through self-attention. Learned
    semantic queries then repeatedly attend to those selected tokens and are
    pooled into the slow-branch semantic anchor h_s.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_keyframes: int = 5,
        token_layers: int = 2,
        class_layers: int = 2,
        num_queries: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        pooling: str = "attention",
        use_residual_gate: bool = True,
    ):
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be >= 1.")
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be 'attention' or 'mean'.")

        self.num_queries = int(num_queries)
        self.pooling = pooling
        self.pos_embed = nn.Parameter(torch.zeros(1, num_keyframes, d_model))
        self.semantic_queries = nn.Parameter(torch.zeros(1, self.num_queries, d_model))

        if token_layers > 0:
            token_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=token_layers)
        else:
            self.token_encoder = nn.Identity()

        self.class_blocks = nn.ModuleList(
            [
                ClassAttentionResidualBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_residual_gate=use_residual_gate,
                )
                for _ in range(max(1, class_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.query_pool = AttentionPooling(d_model) if pooling == "attention" and self.num_queries > 1 else None

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.semantic_queries, std=0.02)

    def forward(
        self,
        selected: torch.Tensor,
        full: torch.Tensor | None = None,
        selected_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del full, selected_indices
        if selected.ndim != 3:
            raise ValueError(f"Expected selected [B, K, D], got {tuple(selected.shape)}")
        if selected.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(
                f"selected length K={selected.shape[1]} exceeds configured num_keyframes={self.pos_embed.shape[1]}"
            )

        b = selected.shape[0]
        tokens = selected + self.pos_embed[:, : selected.shape[1]]
        tokens = self.token_encoder(tokens)

        query = self.semantic_queries.expand(b, -1, -1)
        for block in self.class_blocks:
            query = block(query, tokens)
        query = self.norm(query)

        if self.num_queries == 1:
            return query[:, 0]
        if self.query_pool is not None:
            return self.query_pool(query)
        return query.mean(dim=1)


class SelectedTransformerAttentionPoolingEncoder(nn.Module):
    """Selected-token Transformer with learned attention pooling.

    This is kept as an explicit ablation for "without class-attention
    queries"; the mainline slow branch remains SelectedClassAttentionEncoder.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_keyframes: int = 5,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        pooling: str = "attention",
    ):
        super().__init__()
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be 'attention' or 'mean'.")
        self.pooling = pooling
        self.pos_embed = nn.Parameter(torch.zeros(1, num_keyframes, d_model))
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            self.encoder = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = AttentionPooling(d_model) if pooling == "attention" else None
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        selected: torch.Tensor,
        full: torch.Tensor | None = None,
        selected_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del full, selected_indices
        if selected.ndim != 3:
            raise ValueError(f"Expected selected [B, K, D], got {tuple(selected.shape)}")
        if selected.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(
                f"selected length K={selected.shape[1]} exceeds configured num_keyframes={self.pos_embed.shape[1]}"
            )
        tokens = selected + self.pos_embed[:, : selected.shape[1]]
        tokens = self.norm(self.encoder(tokens))
        if self.attn_pool is not None:
            return self.attn_pool(tokens)
        return tokens.mean(dim=1)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_slow_encoder(model_cfg: dict, ablation_cfg: dict | None = None) -> nn.Module:
    """Build the slow encoder.

    The mainline is selected_class_attention. transformer_attention_pooling is
    retained only for E22-style ablation without semantic class queries.
    """

    ablation_cfg = ablation_cfg or {}
    slow_cfg = dict(model_cfg.get("slow_encoder", {}))
    slow_type = slow_cfg.get("type") or ablation_cfg.get("slow_encoder_type") or "selected_class_attention"
    if slow_type == "transformer_attention_pooling":
        return SelectedTransformerAttentionPoolingEncoder(
            d_model=int(model_cfg.get("hidden_dim", 768)),
            num_keyframes=int(model_cfg.get("num_keyframes", 5)),
            depth=int(slow_cfg.get("depth", 4)),
            num_heads=int(slow_cfg.get("num_heads", 8)),
            mlp_ratio=float(slow_cfg.get("mlp_ratio", 2.0)),
            dropout=float(slow_cfg.get("dropout", 0.1)),
            pooling=slow_cfg.get("pooling", "attention"),
        )
    if slow_type != "selected_class_attention":
        raise ValueError(
            "Legacy slow encoder variants were removed from the runtime after E14. "
            "Use model.slow_encoder.type=selected_class_attention, or transformer_attention_pooling for ablation."
        )

    return SelectedClassAttentionEncoder(
        d_model=int(model_cfg.get("hidden_dim", 768)),
        num_keyframes=int(model_cfg.get("num_keyframes", 5)),
        token_layers=int(slow_cfg.get("token_layers", 2)),
        class_layers=int(slow_cfg.get("class_layers", 2)),
        num_queries=int(slow_cfg.get("num_queries", 1)),
        num_heads=int(slow_cfg.get("num_heads", 8)),
        mlp_ratio=float(slow_cfg.get("mlp_ratio", 2.0)),
        dropout=float(slow_cfg.get("dropout", 0.1)),
        pooling=slow_cfg.get("pooling", "attention"),
        use_residual_gate=_as_bool(slow_cfg.get("use_residual_gate", True)),
    )
