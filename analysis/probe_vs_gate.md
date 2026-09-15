# Pre-action probe vs post-hoc surprise gate on K2 Horizon

Same attempts, per model size. Label: ok == False (the verifier rejected the attempt). AUROC is rank-based;
CIs are percentile bootstraps over attempts (2000 resamples). The gate scores each attempt's own output by
teacher forcing (post-hoc, arXiv 2609.05274 style); the probe reads the residual stream at the last prompt
position before anything is generated (pre-action), with leave-one-run-out cross-validation (folds = run_id;
runs of the same task appear on both sides of a fold, by design).

## k2-0.9b: n = 137 attempts (52 rejected) from 19 runs

| signal | when | AUROC for ok == False [95% CI] | sign flipped (1 - AUROC) |
|---|---|---|---|
| gate self-surprise: mean NLL (n=129) | post-hoc | 0.355 [0.256, 0.456] | 0.645 |
| gate self-surprise: max NLL (n=129) | post-hoc | 0.415 [0.305, 0.525] | 0.585 |
| gate self-surprise: mean_action NLL (n=124) | post-hoc | 0.308 [0.223, 0.404] **(best gate feature, low surprise rejects)** | 0.692 |
| gate self-surprise: mean_think NLL (n=6) | post-hoc | 1.000 [1.000, 1.000] | 0.000 |
| gate scored by 3.7b: mean NLL (n=129) | post-hoc | 0.624 [0.525, 0.723] | 0.376 |
| gate scored by 3.7b: max NLL (n=129) | post-hoc | 0.540 [0.435, 0.645] | 0.460 |
| gate scored by 3.7b: mean_action NLL (n=124) | post-hoc | 0.658 [0.557, 0.752] | 0.342 |
| gate scored by 3.7b: mean_think NLL (n=6) | post-hoc | 1.000 [1.000, 1.000] | 0.000 |
| probe, best layer 12 of 29 (LORO-CV, optimistic: best-of-29) | pre-action | 0.844 [0.775, 0.902] | |
| probe, median over layers | pre-action | 0.792 | |
| probe, middle (layer 14) | pre-action | 0.811 [0.737, 0.877] | |
| probe, last block (layer 27) | pre-action | 0.817 [0.744, 0.880] | |
| probe, final norm (layer 28) | pre-action | 0.820 [0.749, 0.883] | |
| probe, mean of last 64 prompt tokens at layer 12 | pre-action | 0.811 [0.741, 0.877] | |
| baseline: prompt length | pre-action | 0.667 [0.572, 0.757] | |
| baseline: step index | pre-action | 0.632 [0.536, 0.720] | |

Probe: standardized features + L2 logistic regression, C=1.0; at the best layer, C=0.01: 0.837, C=0.1: 0.853, C=1.0: 0.844, C=10.0: 0.834.
Per-layer AUROC: 0:0.78 1:0.79 2:0.79 3:0.79 4:0.78 5:0.77 6:0.76 7:0.77 8:0.80 9:0.80 10:0.79 11:0.80 12:0.84 13:0.83 14:0.81 15:0.82 16:0.79 17:0.79 18:0.76 19:0.77 20:0.78 21:0.79 22:0.78 23:0.79 24:0.79 25:0.80 26:0.81 27:0.82 28:0.82.

By verifier reason (self-surprise gate vs probe, medians):

| reason | n | median mean NLL | median max NLL | median probe P(rejected) |
|---|---|---|---|---|
| ok | 75 | 0.165 | 1.93 | 0.00 |
| repeat_call | 38 | 0.087 | 0.93 | 0.75 |
| judge_no | 8 | 0.325 | 7.06 | 0.38 |
| final | 5 | 0.804 | 5.86 | 0.00 |
| unknown_tool:edit | 2 | 0.181 | 7.61 | 0.98 |
| unknown_tool:edit_file | 1 | 0.117 | 8.01 | 0.97 |

Spearman between the probe's out-of-fold P(rejected) (best layer) and the gate's surprise, same attempts:

| gate signal | n | Spearman rho [95% CI] | rho on accepted attempts only |
|---|---|---|---|
| self mean NLL | 129 | -0.169 [-0.336, -0.003] | +0.024 |
| self max NLL | 129 | -0.026 [-0.209, +0.157] | +0.143 |
| self mean_action NLL | 124 | -0.064 [-0.244, +0.113] | +0.231 |
| by 3.7b mean NLL | 129 | +0.147 [-0.045, +0.340] | +0.109 |
| by 3.7b max NLL | 129 | +0.150 [-0.033, +0.317] | +0.273 |
| by 3.7b mean_action NLL | 124 | +0.213 [+0.019, +0.390] | +0.204 |

Same-attempt check (joined n=129): gate best feature `mean_action` AUROC 0.308 [0.223, 0.404] vs probe best layer AUROC 0.834 [0.765, 0.898].

Throughput: gate 0.9b: 2343 tok/s (187851 prompt+target tokens, peak 2.0 GB); 3.7b: 591 tok/s (170742 prompt+target tokens, peak 6.8 GB). Probe capture 1483 tok/s (186710 prompt tokens, peak 1.9 GB); probe fit 26 s.

Reasons (joined attempts): ok x75, repeat_call x38, judge_no x8, final x5, unknown_tool:edit x2, unknown_tool:edit_file x1.

## k2-3.7b: n = 127 attempts (19 rejected) from 26 runs

| signal | when | AUROC for ok == False [95% CI] | sign flipped (1 - AUROC) |
|---|---|---|---|
| gate self-surprise: mean NLL (n=127) | post-hoc | 0.139 [0.060, 0.237] **(best gate feature, low surprise rejects)** | 0.861 |
| gate self-surprise: max NLL (n=127) | post-hoc | 0.289 [0.173, 0.423] | 0.711 |
| gate self-surprise: mean_action NLL (n=112) | post-hoc | 0.275 [0.162, 0.397] | 0.725 |
| gate self-surprise: mean_think NLL (n=66) | post-hoc | 0.398 [0.108, 0.692] | 0.602 |
| probe, best layer 16 of 37 (LORO-CV, optimistic: best-of-37) | pre-action | 0.881 [0.788, 0.955] | |
| probe, median over layers | pre-action | 0.855 | |
| probe, middle (layer 18) | pre-action | 0.880 [0.777, 0.953] | |
| probe, last block (layer 35) | pre-action | 0.839 [0.732, 0.928] | |
| probe, final norm (layer 36) | pre-action | 0.818 [0.686, 0.922] | |
| probe, mean of last 64 prompt tokens at layer 16 | pre-action | 0.829 [0.673, 0.945] | |
| baseline: prompt length | pre-action | 0.772 [0.641, 0.887] | |
| baseline: step index | pre-action | 0.710 [0.563, 0.839] | |

Probe: standardized features + L2 logistic regression, C=1.0; at the best layer, C=0.01: 0.872, C=0.1: 0.878, C=1.0: 0.881, C=10.0: 0.884.
Per-layer AUROC: 0:0.77 1:0.80 2:0.85 3:0.84 4:0.84 5:0.83 6:0.84 7:0.85 8:0.84 9:0.84 10:0.84 11:0.85 12:0.86 13:0.85 14:0.87 15:0.88 16:0.88 17:0.87 18:0.88 19:0.88 20:0.88 21:0.88 22:0.87 23:0.86 24:0.84 25:0.86 26:0.88 27:0.88 28:0.87 29:0.86 30:0.87 31:0.87 32:0.86 33:0.85 34:0.85 35:0.84 36:0.82.

By verifier reason (self-surprise gate vs probe, medians):

| reason | n | median mean NLL | median max NLL | median probe P(rejected) |
|---|---|---|---|---|
| ok | 94 | 0.103 | 1.72 | 0.00 |
| final | 14 | 0.305 | 2.61 | 0.00 |
| repeat_call | 9 | 0.019 | 0.37 | 0.55 |
| judge_no | 8 | 0.029 | 1.88 | 0.10 |
| too_many_calls | 1 | 0.062 | 1.27 | 0.00 |
| parse:unclosed_think | 1 | 0.217 | 3.19 | 0.00 |

Spearman between the probe's out-of-fold P(rejected) (best layer) and the gate's surprise, same attempts:

| gate signal | n | Spearman rho [95% CI] | rho on accepted attempts only |
|---|---|---|---|
| self mean NLL | 127 | -0.464 [-0.596, -0.312] | -0.286 |
| self max NLL | 127 | -0.210 [-0.362, -0.045] | -0.076 |
| self mean_action NLL | 112 | -0.277 [-0.465, -0.084] | -0.185 |

Same-attempt check (joined n=127): gate best feature `mean` AUROC 0.139 [0.060, 0.237] vs probe best layer AUROC 0.881 [0.788, 0.955].

Throughput: gate 3.7b: 579 tok/s (220311 prompt+target tokens, peak 6.6 GB). Probe capture 136 tok/s (203655 prompt tokens, peak 6.3 GB); probe fit 78 s.

Reasons (joined attempts): ok x94, final x14, repeat_call x9, judge_no x8, too_many_calls x1, parse:unclosed_think x1.
