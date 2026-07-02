# RESULTS v2 — query_concat 3-way ablation on ramen

**Date**: 2026-05-18
**Scope**: ramen scene only (no Kendall this round per request)
**New arch**: `--unctoken_arch query_concat` — f_u as N+1 query concatenated to Q
side of the main cross-attention, then routed to PresentHead via three
different paths controlled by `--qc_literal` / `--qc_no_layernorm`.

## 1. Experimental design

Shared hyperparameters across all 3 lanes (and shared with v1 ramen baselines):
- `--total_iters 45000` (resume from `ramenchkpnt30000.pth`)
- `--training_neg_variants attribute,category,spatial,borrow`
- `--training_neg_target_ratio 0.20`
- `--lambda_com 0.1`, `--lambda_classifier 1.0`
- `--use_present_head --use_uncertain_token --unctoken_arch query_concat`

Three lanes differ ONLY in how `f_u_output` from cross-attention feeds the
PresentHead FFN:

| Lane | Extra flags | PresentHead input | Scene-aware? |
|---|---|---|---|
| **uc_qc** | (default) | `qc_fusion(concat(g_global, LN(f_u_output)))` | ✓ (g_global) |
| **uc_qc_literal** | `--qc_literal` | `LN(f_u_output)` | ✗ |
| **uc_qc_raw** | `--qc_literal --qc_no_layernorm` | `f_u_output` (raw, professor) | ✗ |

Evaluation: per-variant `--test_neg_target_ratio 0.20` (matches training distribution),
plus diagnostic CSVs at both `neg020` and `native` sweeps for cutoff-free AUROC / mIoU+.

## 2. Main results — neg020 sweep (matches training distribution)

Per-variant table (lower-half = new this round):

| Method | variant | n_pos | n_neg | AUROC | PR-AUC | N@0 | T@0 | mIoU+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base          | attribute | 52 | 13 | 0.519 | 0.818 | 0.385 | 0.731 | 0.115 |
| base          | borrow    | 44 | 11 | 0.944 | 0.987 | 1.000 | 0.727 | 0.139 |
| base          | category  | 32 | 8  | 0.723 | 0.936 | 0.000 | 0.781 | 0.129 |
| base          | spatial   | 52 | 13 | 0.932 | 0.983 | 1.000 | 0.731 | 0.115 |
| uc (inline)   | attribute | 52 | 13 | 0.704 | 0.913 | 0.308 | 0.769 | 0.176 |
| uc (inline)   | borrow    | 44 | 11 | 0.926 | 0.982 | 1.000 | 0.727 | 0.185 |
| uc (inline)   | category  | 32 | 8  | 0.875 | 0.974 | 1.000 | 0.844 | 0.194 |
| uc (inline)   | spatial   | 52 | 13 | 0.941 | 0.985 | 1.000 | 0.769 | 0.176 |
| uck_self      | attribute | 52 | 13 | 0.522 | 0.834 | 0.231 | 0.808 | 0.195 |
| uck_self      | borrow    | 44 | 11 | 0.979 | 0.995 | 1.000 | 0.818 | 0.231 |
| uck_self      | category  | 32 | 8  | 0.855 | 0.953 | 0.875 | 0.875 | 0.243 |
| uck_self      | spatial   | 52 | 13 | 0.925 | 0.982 | 1.000 | 0.808 | 0.195 |
| **uc_qc**         | attribute | 52 | 13 | 0.621 | 0.871 | 0.231 | 1.000 | 0.160 |
| **uc_qc**         | borrow    | 44 | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 0.175 |
| **uc_qc**         | category  | 32 | 8  | 0.945 | 0.988 | 0.125 | 1.000 | 0.171 |
| **uc_qc**         | spatial   | 52 | 13 | 0.873 | 0.948 | 0.692 | 1.000 | 0.160 |
| **uc_qc_literal** | attribute | 52 | 13 | 0.466 | 0.802 | 0.231 | 1.000 | 0.165 |
| **uc_qc_literal** | borrow    | 44 | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 0.188 |
| **uc_qc_literal** | category  | 32 | 8  | **0.277** | 0.770 | 0.000 | 1.000 | 0.197 |
| **uc_qc_literal** | spatial   | 52 | 13 | 0.923 | 0.975 | 0.846 | 1.000 | 0.165 |
| **uc_qc_raw**     | attribute | 52 | 13 | 0.481 | 0.821 | 0.231 | 0.904 | 0.155 |
| **uc_qc_raw**     | borrow    | 44 | 11 | 1.000 | 1.000 | 1.000 | 0.909 | 0.173 |
| **uc_qc_raw**     | category  | 32 | 8  | 0.922 | 0.983 | 0.875 | 0.938 | 0.163 |
| **uc_qc_raw**     | spatial   | 52 | 13 | 0.886 | 0.963 | 0.692 | 0.904 | 0.155 |

### 4-variant means (the headline numbers)

| Method | AUROC | PR-AUC | T@0 | mIoU+ |
|---|---:|---:|---:|---:|
| base | 0.779 | 0.931 | 0.743 | 0.124 |
| uc (inline) | 0.861 | 0.964 | 0.777 | **0.183** |
| uck_self (inline + Kendall) | 0.820 | 0.941 | 0.827 | **0.216** |
| **uc_qc** (fusion + LN) | **0.860** | 0.952 | **1.000** | 0.167 |
| **uc_qc_literal** (LN only) | 0.667 ⚠️ | 0.887 | **1.000** | 0.179 |
| **uc_qc_raw** (professor 字面) | 0.822 | 0.942 | 0.914 | 0.162 |

## 3. Native sweep (no neg ratio enforcement) — sanity check

| Method | AUROC | mIoU+ |
|---|---:|---:|
| base | 0.766 | 0.100 |
| uc (inline) | 0.848 | 0.135 |
| uck_self | 0.813 | 0.187 |
| **uc_qc** | 0.838 | 0.148 |
| **uc_qc_literal** | 0.657 | 0.162 |
| **uc_qc_raw** | 0.815 | 0.148 |

Same ordering as neg020. **uc_qc_literal collapses on category in both sweeps**
(0.277 / 0.230), confirming this is not an eval-distribution artifact.

## 4. Answers to professor's questions

### Q1: query_concat 比 inline uc 好吗？

**几乎打平，方向不一致**：
- AUROC: uc_qc 0.860 vs uc (inline) 0.861 — 持平 (Δ = −0.001)
- mIoU+: uc_qc 0.167 vs uc (inline) 0.183 — inline 略好 (Δ = −0.016)
- T@0: uc_qc 1.000 vs inline 0.777 — query_concat 在默认阈值更 confident（PresentHead
  logit 整体偏高，所有正样本都被预测为 present）

**结论**: 不构成对 inline 的明显胜出。考虑到 uc_qc 多了一个 N+1 query + LN + 2D→D
fusion 投影，没有"免费的午餐"。

### Q2: g_global fusion 重要吗？

**非常重要**：
- uc_qc (有 g_global fusion) AUROC = 0.860
- uc_qc_literal (无 fusion, 仅 LN) AUROC = 0.667
- **Δ = +0.19 AUROC** 来自加入 g_global

这印证了 codex review #9 的担心: 单看 `f_u_output` 是 sentence-aware 但 *scene-blind* 的——
attention 的输出主要被 word sequence 主导，缺少 "本场景里这些 Gaussian 整体状况"
的信号。融入 opacity-weighted `g_global` 把这个空缺补上了。

### Q3: LayerNorm 重要吗？

**反直觉地伤害了 AUROC**：
- uc_qc_raw (无 LN) AUROC = 0.822
- uc_qc_literal (有 LN) AUROC = 0.667
- **Δ = −0.155 AUROC** 来自加 LN

最戏剧的是 **category 变体**：raw=0.922 vs literal=0.277（literal 比随机还差 0.5）。
这说明在没有 g_global 的情况下，对 `f_u_output` 做 LN 会破坏类间区分性
（推测 LN 把 scale 压平后，PresentHead 失去了 logit 量级这一关键判断维度）。

**注意**：当同时有 g_global fusion 时（uc_qc），LN 不再有害——因为 g_global
本身保留了 scale 信息，下游 `qc_fusion` Linear 还能学回正确投影。

### Q4: Professor 字面版 (uc_qc_raw) 能开箱即用吗？

**勉强可用但不推荐**：
- AUROC 0.822 > baseline 0.779（+0.043），有一定改善
- mIoU+ 0.162 ≈ baseline 0.124（+0.038），改善小
- 但相比 uc inline (0.861 / 0.183) 没有任何优势
- 相比 uc_qc fusion 版 (0.860 / 0.167) AUROC 落后 0.038

**核心信号**：professor 的"f_u 作为 N+1 query 拼到 Q 侧"思路是对的，但"直接拿
`f_u_output` 喂 PresentHead"这一字面实现缺少场景上下文。codex 建议的
`qc_fusion(concat(g_global, ...))` 补回这个上下文，把 AUROC 从 0.822 提到 0.860。

## 5. 整体结论

| 排序（AUROC） | 方法 | 设计点 |
|---|---|---|
| 1️⃣ 0.861 | uc (inline)        | f_u 拼到 word 序列里，参与联合 softmax |
| 2️⃣ 0.860 | **uc_qc**          | f_u 作为 N+1 query + g_global fusion |
| 3️⃣ 0.822 | **uc_qc_raw**      | f_u 作为 N+1 query 直接喂 PresentHead |
| 4️⃣ 0.820 | uck_self           | inline + Kendall σ²-self |
| 5️⃣ 0.779 | base               | 仅 PresentHead，无 f_u |
| 6️⃣ 0.667 | **uc_qc_literal**  | 同 3 但加 LN（伤害 category） |

**排序（mIoU+）**：uck_self (0.216) > uc inline (0.183) > uc_qc_literal (0.179) >
uc_qc (0.167) > uc_qc_raw (0.162) > base (0.124)
→ Kendall σ²-self 仍是 mIoU+ 最高的（v1 已记录），query_concat 系列在 mIoU+ 上
都不如 inline。

## 6. 复现 / 文件

| 文件 | 路径 |
|---|---|
| Orchestrator | `output/train_ramen_qc_3way.sh` |
| Re-eval (post-fix) | `output/reeval_ramen_qc_raw_literal.sh` |
| 分析脚本 | `output/analyze_diag_csvs.py` |
| 输出目录 | `output/ramen_uc_qc{,_literal,_raw}` |
| 关键 commit | `f697e60` (qc_no_layernorm flag) |

### 修复记录

第一轮 eval（auto-triggered after training）失败：`test_metrics.py` 不识别
`--qc_literal` / `--qc_no_layernorm`。修复方法：在 `test_metrics.py:288-298`
增加 argparse 注册，并把两个 flag 传入 `GaussianModel(...)` 构造函数
（line 36-44）。修复后 `reeval_ramen_qc_raw_literal.sh` 重跑 eval+diag CSVs
通过。
