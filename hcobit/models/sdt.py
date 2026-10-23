from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import config_dict

# ============================================================
#  RoPE utilities (SDPA-friendly, compile/AMP safe)
# ============================================================


class RotaryEmbedding(nn.Module):
    """Stateless RoPE. Computes cos/sin for a given seq_len (float32 internally)."""

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ============================================================
#  Attention wrapper (FLASH when possible, safe under compile/AMP)
# ============================================================


class SelfAttention(nn.Module):
    """
    Principled attention: one parameterization, multiple kernels.

    - Uses PyTorch SDPA (FlashAttention kernels when possible).
    - Supports self-attn and cross-attn.
    - Supports key_padding_mask (True = pad/ignore) and attn_bias.
    - Optional RoPE for SELF-attn.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        *,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"
        self.head_dim = self.d_model // self.nhead
        self.cross_attn = bool(cross_attn)
        self.dropout = float(dropout)

        if self.cross_attn:
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
            self.kv_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=True)
            self.qkv_proj = None
        else:
            self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
            self.q_proj = None
            self.kv_proj = None

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        self.use_rope = bool(use_rope) and (not self.cross_attn)
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"use_rope=True requires even head_dim, got head_dim={self.head_dim} "
                    f"(d_model={self.d_model}, nhead={self.nhead})."
                )
            self.rope = RotaryEmbedding(self.head_dim, base=float(rope_base))
        else:
            self.rope = None

        for m in (self.qkv_proj, self.q_proj, self.kv_proj, self.out_proj):
            if m is None:
                continue
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, e = x.shape
        return x.view(b, n, self.nhead, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * dh)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv_src = x_kv if (self.cross_attn and x_kv is not None) else x

        if self.cross_attn:
            q = self.q_proj(x)
            kv = self.kv_proj(kv_src)
            k, v = kv.chunk(2, dim=-1)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rope is not None:
            l = q.shape[-2]
            cos, sin = self.rope(l, device=q.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_mask = kpm

        if attn_bias is not None:
            bias = attn_bias.to(device=q.device, dtype=q.dtype)
            if bias.dim() == 2:
                bias = bias[None, None, :, :]
            if attn_mask is None:
                attn_mask = bias
            else:
                if attn_mask.dtype == torch.bool:
                    neg_inf = torch.finfo(q.dtype).min
                    attn_mask = attn_mask.to(q.dtype) * neg_inf + bias
                else:
                    attn_mask = attn_mask + bias

        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        y = self._merge_heads(y)
        y = self.out_proj(y)
        return y


# ============================================================
#  Helpers: time embedding, blocks, local mixers, heads
# ============================================================


def _build_sinusoidal_table(e: int, device: str):
    inv = torch.exp(-math.log(10_000) * torch.arange(0, e, 2, device=device) / e)
    table = torch.zeros(1, e, device=device)
    table[:, 0::2] = inv
    table[:, 1::2] = inv
    return table


class _SinTimeSigma(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("_freq", _build_sinusoidal_table(dim, "cpu"))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        freq = self._freq.to(sigma.device)
        phases = sigma.log()[:, None] * freq
        emb = torch.empty_like(phases)
        emb[:, 0::2] = phases[:, 0::2].cos()
        emb[:, 1::2] = phases[:, 1::2].sin()
        return emb


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.Wa = nn.Linear(dim, hidden, bias=True)
        self.Wb = nn.Linear(dim, hidden, bias=True)
        self.Wo = nn.Linear(hidden, dim, bias=True)
        self.drop = nn.Dropout(dropout)
        for m in (self.Wa, self.Wb, self.Wo):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.Wa(x)
        b = F.silu(self.Wb(x))
        y = a * b
        y = self.drop(y)
        return self.Wo(y)


class AdaLNZero(nn.Module):
    """Produces (scale, shift, gate) from time embedding; zero-init => identity start."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d))
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor):
        orig_dtype = h.dtype
        s, b, g = self.mlp(t_emb).chunk(3, dim=-1)
        h_norm = F.layer_norm(h, h.shape[-1:]).to(orig_dtype)
        h_mod = (1 + s).unsqueeze(1) * h_norm + b.unsqueeze(1)
        return h_mod, g


class PreNormBlockAda(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = False,
        *,
        use_flash_attn: bool = False,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.dim_ff = int(dim_ff or (4 * d_model))
        self.attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            cross_attn=False,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        if use_swiglu:
            hidden = int(2 * self.dim_ff / 3)
            self.ff = SwiGLU(d_model, hidden, dropout=dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, self.dim_ff),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.dim_ff, d_model),
            )
        self.drop = nn.Dropout(dropout)
        self.adaln1 = AdaLNZero(d_model)
        self.adaln2 = AdaLNZero(d_model)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, gate = self.adaln1(x, t_emb)
        y = self.attn(h, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = x + self.drop(y) * gate.unsqueeze(1)

        h, gate = self.adaln2(x, t_emb)
        y = self.ff(h)
        x = x + self.drop(y) * gate.unsqueeze(1)
        return x


class OptimalSkipMLPHead(nn.Module):
    """
    Compact head used for bit-space models.

    Supports arbitrary noisy_dim, so continuous-bit self-conditioning can pass
    concatenated [noisy_embed || selfcond_embed] while keeping x_denoised at content_dim.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        out_dim: int,
        content_dim: int,
        noisy_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.P = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.content_dim = int(content_dim)
        self.noisy_dim = int(noisy_dim)

        self.patch_adapter = nn.Linear(d_model, self.P * self.hidden_dim)
        self.input_adapter = nn.Linear(self.content_dim + self.noisy_dim, self.hidden_dim)
        self.time_proj = nn.Linear(d_model, self.hidden_dim)
        self.adaln = AdaLNZero(self.hidden_dim)
        self.mixer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        nn.init.xavier_uniform_(self.patch_adapter.weight)
        nn.init.zeros_(self.patch_adapter.bias)
        nn.init.xavier_uniform_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)
        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)
        nn.init.zeros_(self.time_proj.bias)
        for m in self.mixer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_denoised: torch.Tensor,
        x_noisy: torch.Tensor,
        patch_tokens: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = x_denoised.shape
        b2, n, _ = patch_tokens.shape
        if b != b2:
            raise RuntimeError("Batch size mismatch between x_denoised and patch_tokens")

        global_flat = self.patch_adapter(patch_tokens)
        global_feat = global_flat.view(b, n * self.P, self.hidden_dim)
        if global_feat.shape[1] > s:
            global_feat = global_feat[:, :s, :]
        elif global_feat.shape[1] < s:
            global_feat = F.pad(global_feat, (0, 0, 0, s - global_feat.shape[1]))

        local_input = torch.cat([x_denoised, x_noisy], dim=-1)
        local_feat = self.input_adapter(local_input)
        h = global_feat + local_feat

        t_emb_proj = self.time_proj(t_emb)
        h_norm, gate = self.adaln(h, t_emb_proj)
        h_gated = h_norm * gate.unsqueeze(1)
        logits = self.mixer(h_gated)
        return logits


# ============================================================
#   MAIN MODEL: SequenceVDTContinuousModel
# ============================================================


class SequenceVDTContinuousModel(nn.Module):
    """CoBit's Diffusion Transformer over bit patches (one patch = one token's bits).

    Per bit: the noisy bit and the self-conditioning estimate, each scaled by
    c_in = 1 / sqrt(sigma^2 + sigma_data^2) and embedded (content), plus Fourier
    position-within-patch features. A patch is projected to one trunk token; the trunk is
    a stack of AdaLN-Zero transformer blocks conditioned on sigma; the head
    (OptimalSkipMLPHead) maps each token back to per-bit logits, the matched-filter term
    being added outside the model (diffusion/continuous/logit_postprocess.py).

    ``sigma`` [B] (token-bit sigma) drives the time embedding; ``sigma_pos`` [B, S]
    (per-modality schedule) drives c_in per position when given.
    """

    def __init__(self, cfg: config_dict.ConfigDict):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        if str(getattr(m, "head_type", "optimal_skip_mlp")).lower() != "optimal_skip_mlp":
            raise ValueError("only head_type='optimal_skip_mlp' is supported")
        self.P, self.E, self.out_dim = int(m.patch_size), int(m.embed_dim), int(getattr(m, "out_dim", 1))
        dropout = float(getattr(m, "dropout", 0.0))
        self.self_condition = bool(getattr(m, "self_condition", False))
        self.center_inputs = bool(getattr(m, "center_inputs", True))
        self.abs_pos_mode = str(getattr(m, "abs_pos_mode", "full")).lower()
        self.data_center = float(cfg.diffusion.continuous.data_center)

        # content: noisy bit (+ self-conditioning estimate), each embedded to C channels
        self.C = int(getattr(m, "content_dim_continuous", 1))
        self.C_trunk_in = self.C * (2 if self.self_condition else 1)
        self.cont_input_proj = None if self.C == 1 else nn.Linear(1, self.C)

        self.n_fourier_global = int(getattr(m, "n_fourier_global", 8))
        self.n_fourier_local = int(getattr(m, "n_fourier_local", 4))
        self.pos_dim = 2 * self.n_fourier_global + 2 * self.n_fourier_local + 1
        self.patch_dim_in = (self.C_trunk_in + self.pos_dim) * self.P
        if int(getattr(m, "rpb_max_distance", 1)) > 1:
            raise ValueError("relative position bias is not used (rpb_max_distance must be <= 1)")

        self.time_fn = _SinTimeSigma(self.E)
        self.time_proj = nn.Linear(self.E, self.E, bias=False)
        self.time_cond = nn.Linear(self.E, self.E)
        self.patch_proj = nn.Linear(self.patch_dim_in, self.E)
        self.unpatch_proj_content = nn.Linear(self.E, self.C * self.P)
        if not bool(getattr(m, "use_adaln", True)):
            raise ValueError("only AdaLN-Zero blocks are supported")
        self.blocks = nn.ModuleList([
            PreNormBlockAda(
                self.E, int(m.n_heads), dim_ff=int(getattr(m, "dim_ff", 4 * self.E)), dropout=dropout,
                use_swiglu=bool(getattr(m, "use_swiglu", False)), use_flash_attn=bool(getattr(m, "use_flash_attn", False)),
                use_rope=bool(getattr(m, "use_rope_trunk", False)), rope_base=float(getattr(m, "rope_base", 10_000.0)),
            )
            for _ in range(int(m.n_blocks))
        ])
        self.head = OptimalSkipMLPHead(
            d_model=self.E, patch_size=self.P, out_dim=self.out_dim, content_dim=self.C,
            noisy_dim=self.C_trunk_in, hidden_dim=int(getattr(m, "head_hidden", 256)), dropout=dropout,
        )

        for lin in (self.patch_proj, self.time_cond, self.unpatch_proj_content):
            nn.init.normal_(lin.weight, 0.0, 0.02)
            nn.init.zeros_(lin.bias)
        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)
        if self.cont_input_proj is not None:
            nn.init.normal_(self.cont_input_proj.weight, 0.0, 0.02)
            nn.init.zeros_(self.cont_input_proj.bias)

    def _fourier_pos_feats(self, s: int, device, dtype):
        """[1, S, pos_dim]: global Fourier features (zero in 'local_only' mode), Fourier
        features of the position within the patch, and the distance to the patch centre."""
        t32 = torch.float32
        pi2 = 2.0 * math.pi
        idx = torch.arange(s, device=device, dtype=t32)
        if self.n_fourier_global > 0 and self.abs_pos_mode != "local_only":
            omega_g = pi2 * torch.arange(1, self.n_fourier_global + 1, device=device, dtype=t32) / max(s - 1, 1)
            ss = idx[:, None] * omega_g[None, :]
            global_feats = torch.cat([ss.sin(), ss.cos()], dim=-1)
        else:
            global_feats = torch.zeros(s, 2 * self.n_fourier_global, device=device, dtype=t32)
        within = idx % self.P
        if self.n_fourier_local > 0:
            omega_l = pi2 * torch.arange(1, self.n_fourier_local + 1, device=device, dtype=t32) / max(self.P - 1, 1)
            ll = within[:, None] * omega_l[None, :]
            local_feats = torch.cat([ll.sin(), ll.cos()], dim=-1)
        else:
            local_feats = torch.zeros(s, 0, device=device, dtype=t32)
        center = (self.P - 1) / 2.0
        dist_center = ((within - center).abs() / max(center, 1e-6)).unsqueeze(-1)
        return torch.cat([global_feats, local_feats, dist_center], dim=-1).unsqueeze(0).to(dtype)

    def _embed(self, x: torch.Tensor, c_in: torch.Tensor) -> torch.Tensor:
        """[B,S] -> [B,S,C], with c_in [B] or [B,S]."""
        x_scaled = x * (c_in if c_in.dim() >= 2 else c_in.view(-1, 1))
        return x_scaled.unsqueeze(-1) if self.cont_input_proj is None else self.cont_input_proj(x_scaled.unsqueeze(-1))

    def _content(self, x_t, sigma, x0_hat, sigma_pos=None):
        sigma_data = float(self.cfg.diffusion.continuous.sigma_data)
        c_in_sigma = sigma_pos if sigma_pos is not None else sigma
        c_in = 1.0 / (c_in_sigma.pow(2) + sigma_data ** 2).sqrt()
        x_noisy = x_t.float()
        if self.center_inputs:
            x_noisy = x_noisy - self.data_center
        noisy_emb = self._embed(x_noisy, c_in)
        if not self.self_condition:
            return noisy_emb
        if x0_hat is None:
            sc_emb = self._embed(torch.zeros_like(x_t, dtype=torch.float32), c_in)
        else:
            x_sc = x0_hat.float()
            if self.center_inputs:
                x_sc = x_sc - self.data_center
            sc_emb = self._embed(x_sc, c_in)
        return torch.cat([noisy_emb, sc_emb], dim=-1)

    def forward(self, x_t: torch.Tensor, sigma: torch.Tensor, x0_hat: Optional[torch.Tensor] = None,
                sigma_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_t [B,S] bits, sigma [B] -> bit logits [B,S,1] (before the matched filter)."""
        if x_t.dim() != 2 or sigma.dim() != 1 or x_t.size(0) != sigma.size(0):
            raise RuntimeError("expected x_t [B,S] and sigma [B]")
        b, s = x_t.shape
        if s % self.P:
            raise RuntimeError(f"sequence length {s} is not a multiple of the patch size {self.P}")
        content = self._content(x_t, sigma, x0_hat, sigma_pos=sigma_pos)
        x_aug = torch.cat([content, self._fourier_pos_feats(s, content.device, content.dtype).expand(b, -1, -1)], dim=-1)
        n = s // self.P
        tokens = self.patch_proj(x_aug.view(b, n, self.P * x_aug.size(-1)))
        t_emb = self.time_cond(self.time_proj(self.time_fn(sigma)))
        for blk in self.blocks:
            tokens = blk(tokens, t_emb, key_padding_mask=None, attn_bias=None)
        x_recon = self.unpatch_proj_content(tokens).view(b, s, self.C).contiguous()
        return self.head(x_denoised=x_recon, x_noisy=content.to(x_recon.dtype), patch_tokens=tokens, t_emb=t_emb)
