"""Models: the exact TinyCNN of protocol section 6.1 and the optional transformer.

Architecture and parameter count are asserted against the frozen protocol at
construction, so a config/architecture mismatch fails before any training rather
than being discovered in a result table.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Protocol, predicted_receptive_field


class TinyCNN(nn.Module):
    """Section 6.1 architecture: dilated conv stack, mean++max pool, 2 linears.

    Input is a one-hot (batch, 4, L) RNA tensor in A,C,G,U column order.
    Output is a single logit per sequence; no sigmoid (BCEWithLogitsLoss).
    """

    def __init__(self, cfg: dict, in_length: int | None = None):
        super().__init__()
        cin = int(cfg["in_channels"])
        ch = int(cfg["channels"])
        stack = list(cfg["conv_stack"])
        self.conv = nn.ModuleList()
        prev = cin
        for layer in stack:
            self.conv.append(nn.Conv1d(prev, ch, int(layer["kernel"]),
                                       dilation=int(layer["dilation"]),
                                       padding=int(layer["padding"])))
            prev = ch
        self.act = nn.GELU()
        self.head = nn.Linear(2 * ch, int(cfg["hidden"]))
        self.drop = nn.Dropout(float(cfg["dropout"]))
        self.out = nn.Linear(int(cfg["hidden"]), 1)
        self.in_length = in_length
        assert float(cfg["dropout"]) >= 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 4:
            raise ValueError(f"expected (batch, 4, L) one-hot input, got {tuple(x.shape)}")
        h = x
        for conv in self.conv:
            h = self.act(conv(h))
        pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
        return self.out(self.drop(self.act(self.head(pooled)))).squeeze(-1)

    def receptive_field(self) -> int:
        rf = 1
        for conv in self.conv:
            rf += (conv.kernel_size[0] - 1) * conv.dilation[0]
        return rf


class TinyTransformer(nn.Module):
    """Optional H3 architecture, hard-capped by the protocol."""

    def __init__(self, cfg: dict, in_length: int = 201):
        super().__init__()
        d = int(cfg["d_model"])
        self.embed = nn.Embedding(4, d)
        self.pos = nn.Parameter(torch.zeros(1, in_length, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=int(cfg["n_heads"]),
            dim_feedforward=int(cfg["d_ff"]), dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["n_layers"]))
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.argmax(dim=1)
        h = self.embed(idx) + self.pos[:, : idx.shape[1]]
        h = self.encoder(h).mean(dim=1)
        return self.head(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(model_id: str, protocol: Protocol, in_length: int) -> nn.Module:
    if model_id in ("C0", "C1"):
        cfg = protocol.section("tinycnn")
        model = TinyCNN(cfg, in_length=in_length)
        expected = int(cfg["expected_params"])
        got = count_parameters(model)
        if got != expected:
            raise ValueError(f"TinyCNN parameter count {got} != protocol {expected}")
        rf = model.receptive_field()
        expected_rf = predicted_receptive_field(cfg)
        if rf != expected_rf:
            raise ValueError(f"receptive field {rf} nt != protocol {expected_rf} nt")
        return model
    if model_id.startswith("T"):
        tcfg = protocol.section("transformer")
        if not tcfg.get("enabled", False):
            raise ValueError("transformer is disabled in the frozen protocol")
        model = TinyTransformer(tcfg, in_length=in_length)
        n = count_parameters(model)
        cap = int(tcfg["param_cap"])
        if n > cap:
            raise ValueError(f"transformer has {n} parameters, above the {cap} cap")
        return model
    raise ValueError(f"unknown model id {model_id}")


@torch.no_grad()
def predict_logits(model: nn.Module, tensors: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Eval-mode logits, invariant to batch order and batch size."""
    was_training = model.training
    model.eval()
    outs = [model(tensors[i:i + batch_size]) for i in range(0, len(tensors), batch_size)]
    model.train(was_training)
    return torch.cat(outs) if outs else torch.zeros(0)
