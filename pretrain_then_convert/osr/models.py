"""imfDiT: the improved-MeanFlow DiT (Geng et al., arXiv 2512.02012), ported from the
official PyTorch inference code (github.com/Lyy-iiis/imeanflow, branch `torch`).

Module / parameter names are kept identical to the official implementation so that the
released checkpoints (iMF-{B,M,L,XL}-2.pth) load with `load_state_dict` unchanged.

Changes w.r.t. the official file (all numerically equivalent in fp32):
  * RoPE is implemented with real arithmetic (forward-mode AD friendly; no complex views).
  * attention can run through SDPA (flash) or an explicit einsum path ("math"); the math
    path is required inside forward-mode AD (JVP).
  * forward() can compute only the heads that are needed, and the network can be split
    into `trunk()` + `head()` so the iMF loss can run the JVP through trunk+u-head only.
  * the v-head can be omitted (pure flow-matching models) and created later by copying the
    u-head (`add_v_head_from_u`, used when converting FM -> iMF).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# JVP-efficient linear. PyTorch's forward-AD rule for mm/addmm materializes a zero tangent
# for the weight and runs an extra x @ 0 matmul, which would inflate both the wall-clock
# and the *measured FLOPs* of the iMF JVP by ~1.5x. Inside `jvp_mode()` every TorchLinear
# uses this Function, whose jvp computes only dx @ W^T (weights carry no tangent).
# ----------------------------------------------------------------------------------------
_JVP_MODE = [False]
# FLOP counting forces the explicit attention path so that counts are kernel independent
# ("model FLOPs": flash-attention backward recomputation is not counted).
_ATTN_OVERRIDE = [None]


class jvp_mode:
    def __enter__(self):
        self.prev = _JVP_MODE[0]
        _JVP_MODE[0] = True

    def __exit__(self, *a):
        _JVP_MODE[0] = self.prev


class _JVPLinear(torch.autograd.Function):
    @staticmethod
    def forward(x, w, b):
        return F.linear(x, w, b)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, w, b = inputs
        ctx.save_for_backward(x, w)
        ctx.save_for_forward(w)
        ctx.has_bias = b is not None

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gx = gw = gb = None
        gy = gy.to(w.dtype)
        if ctx.needs_input_grad[0]:
            gx = gy.matmul(w)
        if ctx.needs_input_grad[1]:
            gw = gy.reshape(-1, gy.shape[-1]).t().matmul(x.reshape(-1, x.shape[-1]))
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = gy.reshape(-1, gy.shape[-1]).sum(0)
        return gx, gw, gb

    @staticmethod
    def jvp(ctx, dx, dw, db):
        (w,) = ctx.saved_tensors
        if dx is None:
            return None
        return F.linear(dx.to(w.dtype), w)


# ----------------------------------------------------------------------------------------
# basic layers (names follow the official code: `_flax_linear`, `_flax_embedding`, ...)
# ----------------------------------------------------------------------------------------
class TorchLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init="scaled_variance",
                 init_constant=1.0, bias_init="zeros"):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self._flax_linear = nn.Linear(in_features, out_features, bias=bias)
        if weight_init == "scaled_variance":
            nn.init.normal_(self._flax_linear.weight, std=init_constant / math.sqrt(in_features))
        elif weight_init == "zeros":
            nn.init.zeros_(self._flax_linear.weight)
        else:
            raise ValueError(weight_init)
        if bias:
            nn.init.zeros_(self._flax_linear.bias)

    def forward(self, x):
        if _JVP_MODE[0]:
            w, b = self._flax_linear.weight, self._flax_linear.bias
            if torch.is_autocast_enabled():  # cast like autocast would, then run in that dtype
                dt = torch.get_autocast_dtype("cuda")
                x, w, b = x.to(dt), w.to(dt), (b.to(dt) if b is not None else None)
                with torch.autocast("cuda", enabled=False):
                    return _JVPLinear.apply(x, w, b)
            return _JVPLinear.apply(x, w, b)
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        std = 0.02 if weight_init is None else init_constant / math.sqrt(embedding_dim)
        self._flax_embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self._flax_embedding.weight, std=std)

    def forward(self, x):
        return self._flax_embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features, weight_init="scaled_variance", weight_init_constant=1.0):
        super().__init__()
        kw = dict(bias=False, weight_init=weight_init, init_constant=weight_init_constant)
        self.w1 = TorchLinear(in_features, hidden_features, **kw)
        self.w3 = TorchLinear(in_features, hidden_features, **kw)
        self.w2 = TorchLinear(hidden_features, in_features, **kw)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """Scalar -> sinusoidal(256) -> MLP. Same as official (no x1000 scaling of the input)."""

    def __init__(self, hidden_size, frequency_embedding_size=256, weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        kw = dict(out_features=hidden_size, bias=True, weight_init=weight_init, init_constant=init_constant)
        self.mlp = nn.Sequential(TorchLinear(frequency_embedding_size, **kw), nn.SiLU(), TorchLinear(hidden_size, **kw))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

    @torch.no_grad()
    def freeze_to_constant(self, value):
        """Make the embedder output its current value at `value` for *every* input:
        last linear weight -> 0, bias -> E(value). Used at FM->iMF conversion so the
        converted network is exactly the FM network at the branch point."""
        dev = self.mlp[0]._flax_linear.weight.device
        const = self.forward(torch.full((1,), float(value), device=dev))[0]
        last = self.mlp[2]._flax_linear
        last.weight.zero_()
        last.bias.copy_(const)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        self.embedding_table = TorchEmbedding(num_classes + 1, hidden_size, weight_init, init_constant)

    def forward(self, labels):
        return self.embedding_table(labels)


class PatchEmbedder(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, hidden_size, bias=True):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        fan_in, fan_out = patch_size * patch_size * in_channels, hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj.weight, -limit, limit)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).reshape(B, -1, x.shape[1])


# ----------------------------------------------------------------------------------------
# RoPE (1-D over the whole token sequence, pairs (2i, 2i+1) as in the official complex view)
# ----------------------------------------------------------------------------------------
def precompute_rope_cos_sin(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)  # [T, dim/2]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: [B, T, H, D]; cos/sin: [T, D/2]. Returns fp32."""
    x = x.float()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)


# ----------------------------------------------------------------------------------------
# transformer
# ----------------------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, weight_init="scaled_variance", weight_init_constant=1.0):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        kw = dict(in_features=hidden_size, out_features=hidden_size, bias=False,
                  weight_init=weight_init, init_constant=weight_init_constant)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (TorchLinear(**kw) for _ in range(4))
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, attn="sdpa"):
        attn = _ATTN_OVERRIDE[0] or attn
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if attn == "sdpa":
            dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else v.dtype
            o = F.scaled_dot_product_attention(q.transpose(1, 2).to(dt), k.transpose(1, 2).to(dt),
                                               v.transpose(1, 2).to(dt))
            o = o.transpose(1, 2)
        else:  # explicit math path (forward-mode AD compatible), softmax in fp32 as official
            w = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(self.head_dim), k)
            w = F.softmax(w, dim=-1, dtype=torch.float32)
            o = torch.einsum("bhqk,bkhd->bqhd", w.to(v.dtype), v)
        return self.out_proj(o.reshape(B, T, self.hidden_size))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, weight_init="scaled_variance", weight_init_constant=1.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(hidden_size, num_heads, weight_init, weight_init_constant)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio), weight_init, weight_init_constant)
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, cos, sin, attn="sdpa"):
        x = x + self.attn(self.norm1(x), cos, sin, attn) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(hidden_size, patch_size * patch_size * out_channels, bias=True, weight_init="zeros")

    def forward(self, x):
        return self.linear(self.norm(x))


class imfDiT(nn.Module):
    """Shared trunk of (depth - aux_head_depth) blocks, then a u-head and (optionally) a
    v-head of aux_head_depth blocks each. Conditions only on h = t - r (no explicit t),
    CFG scale omega (via 1 - 1/omega), CFG interval (t_min, t_max) and the class label,
    all through in-context tokens."""

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16,
                 mlp_ratio=8 / 3, num_classes=1000, aux_head_depth=8, num_class_tokens=8, num_time_tokens=4,
                 num_cfg_tokens=4, num_interval_tokens=2, token_init_constant=1.0, embedding_init_constant=1.0,
                 weight_init_constant=0.32, with_v_head=True, eval_mode=False):
        super().__init__()
        self.input_size, self.patch_size, self.in_channels = input_size, patch_size, in_channels
        self.hidden_size, self.depth, self.num_heads = hidden_size, depth, num_heads
        self.num_classes = num_classes
        self.aux_head_depth = aux_head_depth
        self.out_channels = in_channels

        self.x_embedder = PatchEmbedder(input_size, patch_size, in_channels, hidden_size, bias=True)
        ekw = dict(hidden_size=hidden_size, weight_init="scaled_variance", init_constant=embedding_init_constant)
        self.h_embedder = TimestepEmbedder(**ekw)
        self.omega_embedder = TimestepEmbedder(**ekw)
        self.cfg_t_start_embedder = TimestepEmbedder(**ekw)
        self.cfg_t_end_embedder = TimestepEmbedder(**ekw)
        self.y_embedder = LabelEmbedder(num_classes, **ekw)

        std = token_init_constant / math.sqrt(hidden_size)
        self.time_tokens = nn.Parameter(torch.randn(num_time_tokens, hidden_size) * std)
        self.class_tokens = nn.Parameter(torch.randn(num_class_tokens, hidden_size) * std)
        self.omega_tokens = nn.Parameter(torch.randn(num_cfg_tokens, hidden_size) * std)
        self.t_min_tokens = nn.Parameter(torch.randn(num_interval_tokens, hidden_size) * std)
        self.t_max_tokens = nn.Parameter(torch.randn(num_interval_tokens, hidden_size) * std)

        self.prefix_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens + num_time_tokens
        total_tokens = self.x_embedder.num_patches + self.prefix_tokens
        cos, sin = precompute_rope_cos_sin(hidden_size // num_heads, total_tokens)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        bkw = dict(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   weight_init="scaled_variance", weight_init_constant=weight_init_constant)
        self.shared_blocks = nn.ModuleList([TransformerBlock(**bkw) for _ in range(depth - aux_head_depth)])
        self.u_heads = nn.ModuleList([TransformerBlock(**bkw) for _ in range(aux_head_depth)])
        n_v = aux_head_depth if (with_v_head and not eval_mode) else 0
        self.v_heads = nn.ModuleList([TransformerBlock(**bkw) for _ in range(n_v)])
        self.u_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # the official code always builds v_final_layer; pure FM models (no v-head) skip it so
        # that DDP sees no unused parameters. It is created again at FM -> iMF conversion.
        self.v_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels) if n_v > 0 or eval_mode else None

    # ------------------------------------------------------------------
    @property
    def has_v_head(self):
        return len(self.v_heads) > 0

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def build_sequence(self, x, h, w, t_min, t_max, y):
        seq = torch.cat([
            self.class_tokens + self.y_embedder(y)[:, None],
            self.omega_tokens + self.omega_embedder(1 - 1 / w)[:, None],
            self.t_min_tokens + self.cfg_t_start_embedder(t_min)[:, None],
            self.t_max_tokens + self.cfg_t_end_embedder(t_max)[:, None],
            self.time_tokens + self.h_embedder(h)[:, None],
            self.x_embedder(x),
        ], dim=1)
        return seq.float()  # keep the residual stream in fp32 under autocast

    def enable_compile(self, mode=None):
        """torch.compile each transformer block (fuses the fp32 RMSNorm / RoPE / gating
        elementwise ops). Compiled blocks are used everywhere except inside the JVP
        (forward-mode AD runs eager) and inside FLOP counting."""
        self._compile_mode = mode or "default"
        self._compiled = True

    def _run_block(self, blk, seq, attn):
        if getattr(self, "_compiled", False) and not _JVP_MODE[0] and _ATTN_OVERRIDE[0] is None:
            fn = getattr(blk, "_cfwd", None)
            if fn is None:
                fn = torch.compile(blk.forward, mode=self._compile_mode, dynamic=False)
                object.__setattr__(blk, "_cfwd", fn)
            return fn(seq, self.rope_cos, self.rope_sin, attn)
        return blk(seq, self.rope_cos, self.rope_sin, attn)

    def trunk(self, x, h, w, t_min, t_max, y, attn="sdpa"):
        seq = self.build_sequence(x, h, w, t_min, t_max, y)
        for blk in self.shared_blocks:
            seq = self._run_block(blk, seq, attn)
        return seq

    def head(self, seq, which, attn="sdpa"):
        blocks, final = (self.u_heads, self.u_final_layer) if which == "u" else (self.v_heads, self.v_final_layer)
        for blk in blocks:
            seq = self._run_block(blk, seq, attn)
        return self.unpatchify(final(seq[:, self.prefix_tokens:]).float())

    def forward(self, x, t, h, w, t_min, t_max, y, heads=("u", "v"), attn="sdpa"):
        """Official signature. `t` is unused (the model conditions on h = t - r only).
        Returns (u, v); heads not requested are None."""
        seq = self.trunk(x, h, w, t_min, t_max, y, attn)
        u = self.head(seq, "u", attn) if "u" in heads else None
        v = self.head(seq, "v", attn) if ("v" in heads and self.has_v_head) else None
        return u, v

    # ------------------------------------------------------------------
    @torch.no_grad()
    def add_v_head_from_u(self):
        """FM -> iMF conversion: v-head := copy of u-head (so v = u = FM velocity)."""
        self.v_heads = copy.deepcopy(self.u_heads)
        self.v_final_layer = copy.deepcopy(self.u_final_layer)

    @torch.no_grad()
    def freeze_new_condition_embedders(self):
        """FM -> iMF conversion: the FM network only ever saw h=0, omega=1, [t_min,t_max]=[0,1].
        Make those embedders constant at exactly those values (zero last weight), so the
        converted network equals the FM network for every (h, omega, interval) at step 0."""
        self.h_embedder.freeze_to_constant(0.0)
        self.omega_embedder.freeze_to_constant(0.0)  # input is 1 - 1/omega = 0 at omega = 1
        self.cfg_t_start_embedder.freeze_to_constant(0.0)
        self.cfg_t_end_embedder.freeze_to_constant(1.0)


# ----------------------------------------------------------------------------------------
# size ladder. B/M/L/XL are exactly the official iMF configs; XS, S and H extend the ladder.
# ----------------------------------------------------------------------------------------
SIZES = {
    "XS": dict(depth=12, hidden_size=384, num_heads=6),  # below S (not an official size)
    "S": dict(depth=12, hidden_size=512, num_heads=8),
    "B": dict(depth=12, hidden_size=768, num_heads=12),
    "M": dict(depth=24, hidden_size=768, num_heads=12),
    "L": dict(depth=32, hidden_size=1024, num_heads=16),
    "XL": dict(depth=48, hidden_size=1024, num_heads=16),
    "H": dict(depth=48, hidden_size=1536, num_heads=24),
}


def build_model(size, with_v_head=True, eval_mode=False, **kw):
    cfg = dict(SIZES[size])
    cfg.update(kw)
    return imfDiT(patch_size=2, aux_head_depth=8, with_v_head=with_v_head, eval_mode=eval_mode, **cfg)


def count_params(model, inference_only=True):
    """Non-embedding parameters (transformer blocks + final layer of the u-path)."""
    blocks = list(model.shared_blocks) + list(model.u_heads)
    if not inference_only:
        blocks += list(model.v_heads)
    return sum(p.numel() for b in blocks for p in b.parameters())
