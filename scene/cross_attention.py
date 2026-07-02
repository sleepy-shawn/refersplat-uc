import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from uncertainty.present_classifier import GaussianAttrConvPoolFormerHead, PresentHead

class CrossAttention(nn.Module):
    # Mapping of design-doc symbols to code:
    #   f_r,i   = self.q_linear(g)            pre-attention semantic (q-space)
    #   f_p,i   = self.gp_linear(g_p_in)      per-Gaussian position embedding
    #   f'_r,i  = softmax(QK^T) · V           post-PCMI semantic; this IS the
    #                                          per-Gaussian text-aggregated content
    #                                          (in v-space). Sentence-dependent at
    #                                          inference, unlike f_r.
    # Main-attention query is always Q = f_r + f_p (kept fixed across modes so
    # the mask head is comparable across ablation runs). Main-attention output
    # path (residual + LayerNorm) is also unchanged across modes.
    #
    # `unctoken_query_mode` controls ONLY the u_i (uncertain-token) branch:
    #   "fr_plus_fp"     : q_for_u = Q = f_r + f_p
    #                      sentence-INVARIANT at inference (f_r, f_p depend
    #                      only on per-Gaussian state). u_i can shape itself
    #                      during training via the text-driven W_q, but at
    #                      eval time U is identical for any sentence under
    #                      the same view.
    #   "frpost_plus_fp" : q_for_u = f'_r + f_p
    #                      sentence-DEPENDENT at inference via f'_r. Tests
    #                      whether adding sentence-conditioned signal lets
    #                      the gate vary U across sentences (e.g. "left of
    #                      pot" vs "right of pot").
    #   "frpost_only"    : q_for_u = f'_r
    #                      pure post-PCMI text-aggregated content, no
    #                      position. Cleanest "does the gate need position
    #                      at all" test.
    _UNC_QUERY_MODES = ("fr_plus_fp", "frpost_plus_fp", "frpost_only")
    # "external" — original: separate W_q_u/W_k_u, f_u not in attention.
    # "inline"   — CLS-style: f_u concat'd to WORD sequence (K/V side);
    #              Gaussian queries attend to {words + f_u}.
    # "query_concat" — NEW (2026-05-18): f_u concat'd to GAUSSIAN sequence
    #              (Q side) as an extra query. Cross-attention runs over
    #              Q=[g_0..g_{N-1}, f_u_query] vs K/V=word only.
    #              Per-Gaussian g_output is the first N rows (unchanged).
    #              The f_u_output (last row, [D]) is a "scene-level
    #              sentence-conditioned absent feature" → fed to
    #              PresentHead's FFN (replaces g_global as PH input).
    _UNC_ARCHS = ("external", "inline", "query_concat")

    def __init__(self, dim, num_heads, use_uncertain_token=False,
                 unctoken_query_mode="fr_plus_fp", unctoken_arch="external",
                 use_q_nt=False, q_nt_num_queries=1, q_nt_no_fp=False):

        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # 缩放因子
        self.dim = dim
        self.use_uncertain_token = use_uncertain_token
        self.use_q_nt = use_q_nt
        self.q_nt_num_queries = int(q_nt_num_queries)
        self.q_nt_no_fp = bool(q_nt_no_fp)
        if self.q_nt_num_queries < 1:
            raise ValueError("q_nt_num_queries must be >= 1")
        if unctoken_query_mode not in self._UNC_QUERY_MODES:
            raise ValueError(
                f"unctoken_query_mode must be one of {self._UNC_QUERY_MODES}, "
                f"got {unctoken_query_mode!r}"
            )
        if unctoken_arch not in self._UNC_ARCHS:
            raise ValueError(
                f"unctoken_arch must be one of {self._UNC_ARCHS}, got {unctoken_arch!r}"
            )
        self.unctoken_query_mode = unctoken_query_mode
        self.unctoken_arch = unctoken_arch

        self.q_linear = nn.Linear(dim, dim)  # Query
        self.k_linear = nn.Linear(dim, dim)  # Key
        self.v_linear = nn.Linear(dim, dim)  # Value
        self.gp_linear = nn.Linear(dim, dim)
        self.kp_linear = nn.Linear(dim, dim)
        self.norm=nn.LayerNorm(dim)

        if use_q_nt:
            if self.q_nt_no_fp:
                self.f_nt = nn.Parameter(torch.randn(self.q_nt_num_queries, dim) * 0.02)
            else:
                self.f_r_nt = nn.Parameter(torch.randn(self.q_nt_num_queries, dim) * 0.02)
                self.f_p_nt = nn.Parameter(torch.randn(self.q_nt_num_queries, dim) * 0.02)

        if use_uncertain_token:
            self.f_u = nn.Parameter(torch.randn(1, dim) * 0.02)
            if unctoken_arch == "external":
                # f_u 不进 word 序列，旁路 gate 分支用独立 W_q_u/W_k_u
                self.W_q_u = nn.Linear(dim, dim)
                self.W_k_u = nn.Linear(dim, dim)
            elif unctoken_arch == "inline":
                # CLS-style: f_u 拼进 word 序列（K/V 侧）
                # 主 attention 复用 k_linear/v_linear；UCT 在 K/V 里跟 word 一起 softmax
                # per-Gaussian uncertainty 由一个小 MLP 从 "UCT 对该 gaussian 的贡献" 提取
                self.u_mlp = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, 1),
                )
            # "query_concat" branch: f_u as N+1 query attending over words only.
            # Raw f_u_output goes straight to PresentHead — no LN, no fusion.
            # Faithful to professor's spec; ablation found no statistical
            # advantage from qc_norm / qc_fusion add-ons.


    def forward(self, g, g_p, W, return_uncertainty=False, return_q_nt=False):

        W=W.squeeze(0)

        # ----- Branch -1: q_nt no-target query token(s) -----
        # q_nt appends M learnable query rows on the Gaussian-query side.
        # The first N output rows are the standard Gaussian features for the
        # mask head; the M q_nt rows are returned raw for PresentHead.
        if self.use_q_nt:
            k_p_words = torch.matmul(F.softmax(torch.matmul(W, g.transpose(-1, -2)), dim=-1), g_p)
            k_p_words = self.kp_linear(k_p_words)                  # [seq, D]
            f_r = self.q_linear(g)                                  # [N, D]
            f_p = self.gp_linear(g_p)                               # [N, D]
            Q_gauss = f_r + f_p                                     # [N, D]
            if self.q_nt_no_fp:
                Q_nt = self.q_linear(self.f_nt)                    # [M, D]
            else:
                Q_nt = self.q_linear(self.f_r_nt) + self.gp_linear(self.f_p_nt)  # [M, D]
            Q_ext = torch.cat([Q_gauss, Q_nt], dim=0)               # [N+M, D]
            K = self.k_linear(W) + k_p_words                        # [seq, D]
            V = self.v_linear(W)                                    # [seq, D]
            attention_scores = torch.matmul(Q_ext, K.transpose(-1, -2)) * self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)  # [N+M, seq]
            out_ext = torch.matmul(attention_weights, V)             # [N+M, D]
            g_attn = out_ext[:-self.q_nt_num_queries]                # [N, D]
            qnt_output = out_ext[-self.q_nt_num_queries:]            # [M, D]
            output = self.norm(g_attn + g)                           # [N, D]

            if return_q_nt:
                return output, qnt_output
            return output

        # ----- Branch 0: query_concat UCT (NEW, 2026-05-18) -----
        # f_u as an extra query row; Q = [Gaussians; f_u_query].
        # K, V = words only. f_u_output (last row) replaces g_global as
        # PresentHead's FFN input upstream.
        if self.use_uncertain_token and self.unctoken_arch == "query_concat":
            # Build per-Gaussian query and the single f_u query.
            k_p_words = torch.matmul(F.softmax(torch.matmul(W, g.transpose(-1, -2)), dim=-1), g_p)
            k_p_words = self.kp_linear(k_p_words)                  # [seq, D]
            f_r = self.q_linear(g)                                  # [N, D]
            f_p = self.gp_linear(g_p)                               # [N, D]
            Q_gauss = f_r + f_p                                     # [N, D]
            # f_u uses the same q_linear; no position term (it has no xyz).
            f_u_q = self.q_linear(self.f_u)                         # [1, D]
            Q_ext = torch.cat([Q_gauss, f_u_q], dim=0)              # [N+1, D]
            K = self.k_linear(W) + k_p_words                        # [seq, D]
            V = self.v_linear(W)                                    # [seq, D]
            attention_scores = torch.matmul(Q_ext, K.transpose(-1, -2)) * self.scale
            # softmax is per-row over `seq` words. Q rows do NOT interact
            # with each other (Gaussian ↔ Gaussian, Gaussian ↔ f_u: none).
            attention_weights = F.softmax(attention_scores, dim=-1)  # [N+1, seq]
            out_ext = torch.matmul(attention_weights, V)             # [N+1, D]
            g_attn = out_ext[:-1]                                    # [N, D]
            f_u_output = out_ext[-1:]                                # [1, D]
            # Residual + norm only on Gaussian rows (f_u_output is
            # extracted as a separate signal, no residual w.r.t. f_u).
            output = self.norm(g_attn + g)                           # [N, D]

            if return_uncertainty:
                return output, f_u_output                            # raw, [1, D]
            return output

        # ----- Branch 1: inline (CLS-style) UCT -----
        if self.use_uncertain_token and self.unctoken_arch == "inline":
            # Append UCT token to word sequence
            W_ext = torch.cat([W, self.f_u], dim=0)               # [seq+1, D]
            # Word-side position-aware key (k_p) is the per-word "gaussian-position
            # context" — semantically only meaningful for real words. UCT has no
            # inherent position context, so we compute k_p over the original W and
            # append a zero row for the UCT slot.
            k_p_words = torch.matmul(F.softmax(torch.matmul(W, g.transpose(-1, -2)), dim=-1), g_p)
            k_p_words = self.kp_linear(k_p_words)                 # [seq, D]
            k_p_uct = torch.zeros(1, self.dim, device=g.device, dtype=g.dtype)
            k_p = torch.cat([k_p_words, k_p_uct], dim=0)          # [seq+1, D]
            f_r = self.q_linear(g)
            f_p = self.gp_linear(g_p)
            Q = f_r + f_p
            K = self.k_linear(W_ext) + k_p                        # [seq+1, D]
            V = self.v_linear(W_ext)                              # [seq+1, D]
            attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)  # [N, seq+1]
            f_r_post = torch.matmul(attention_weights, V)            # [N, D]
            output = f_r_post + g
            output = self.norm(output)

            if return_uncertainty:
                # Per-Gaussian "UCT contribution": attention weight on UCT × UCT's value vector
                attn_to_uct = attention_weights[:, -1:]              # [N, 1]
                v_uct = V[-1:]                                       # [1, D]
                uct_contribution = attn_to_uct * v_uct               # [N, D] via broadcast
                u = torch.sigmoid(self.u_mlp(uct_contribution))      # [N, 1] in [0,1]
                return output, u
            return output

        # ----- Branch 2: external (original) UCT or no UCT -----
        k_p = torch.matmul(F.softmax(torch.matmul(W, g.transpose(-1, -2)), dim=-1), g_p)
        k_p=self.kp_linear(k_p)
        # Compute f_r and f_p separately so we can ablate the u_i query
        # without touching the main-attention path.
        f_r = self.q_linear(g)        # [N, D] — semantic feature
        f_p = self.gp_linear(g_p)     # [N, D] — position embedding
        Q = f_r + f_p                 # main attention always uses f_r + f_p
        K = self.k_linear(W)+k_p
        V=self.v_linear(W)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # [N, T]
        # f'_r = pure attention output, BEFORE residual + LayerNorm.
        # This is the per-Gaussian text-aggregated content in v-space.
        f_r_post = torch.matmul(attention_weights, V)           # [N, D]
        output = f_r_post

        output=output+g
        output=self.norm(output)

        if return_uncertainty and self.use_uncertain_token:
            # External arch: separate W_q_u / W_k_u branch
            if self.unctoken_query_mode == "frpost_plus_fp":
                q_for_u = f_r_post + f_p                       # text + position
            elif self.unctoken_query_mode == "frpost_only":
                q_for_u = f_r_post                             # pure text
            else:  # "fr_plus_fp" — original, sentence-invariant at inference
                q_for_u = Q
            q_u = self.W_q_u(q_for_u)                          # [N, D]
            k_u = self.W_k_u(self.f_u)                         # [1, D]
            u_logit = (q_u * k_u).sum(dim=-1, keepdim=True)    # [N, 1]
            u_logit = u_logit / math.sqrt(self.dim // self.num_heads)
            u = torch.sigmoid(u_logit)                         # [N, 1] in [0,1]
            return output, u

        return output

class MLP1(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
       
        x = self.fc3(x)
        return x  

class MLP2(nn.Module):
    def __init__(self, in_dim=16, out_dim=128):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32 ,64)
        self.fc3 = nn.Linear(64, 128)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x 
    
class MLP3(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16 ,64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x


class SigmaHead(nn.Module):
    """
    Per-Gaussian aleatoric uncertainty head for TRUE Kendall heteroscedastic
    loss (Kendall & Gal 2017). Outputs log(σ²) per Gaussian so σ² is always
    positive after `exp()`. Trained jointly with mask BCE via:

        L = bce / (2·σ² + ε) + 0.5·log(σ² + ε)

    BOTH terms have gradient → σ² self-emerges to balance them. This is the
    correct implementation of Kendall, in contrast to `use_kendall_aux` which
    computes σ² from cross-view entropy under `torch.no_grad()` and breaks
    the self-emergence mechanism.

    Per-Gaussian σ² is splat to image space via `rasterize_per_gaussian_scalar`
    (same as the deprecated `use_kendall_aux` path).

    Init: weight=0, bias=-1.0 — output starts as log σ² ≈ -1 (σ² ≈ 0.37,
    BCE multiplier ≈ 1.35). Slightly stronger than vanilla BCE at start so
    the mask head learns shape during the warmup period; σ² then drifts
    upward for hard pixels (codex review #3).
    """
    def __init__(self, D=128, init_bias: float = -1.0):
        super().__init__()
        self.linear = nn.Linear(D, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, g):
        # g: (N, D) per-Gaussian features. Output: (N,) log σ² per Gaussian.
        return self.linear(g).squeeze(-1)


class ReferUncertaintyHead(nn.Module):
    """Variational posterior head over 16D refer-feature perturbations."""

    def __init__(self, D=16, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden, D)
        self.log_sigma = nn.Linear(hidden, D)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.zeros_(self.log_sigma.bias)

    def forward(self, x):
        h = self.backbone(x)
        return self.mu(h), self.log_sigma(h)
