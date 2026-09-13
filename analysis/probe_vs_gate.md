# Pre-action probe vs post-hoc surprise gate on K2 Horizon

Same attempts, per model size. Label: ok == False (the verifier rejected the attempt). AUROC is rank-based;
CIs are percentile bootstraps over attempts (2000 resamples). The gate scores each attempt's own output by
teacher forcing (post-hoc, arXiv 2609.05274 style); the probe reads the residual stream at the last prompt
position before anything is generated (pre-action), with leave-one-run-out cross-validation (folds = run_id;
runs of the same task appear on both sides of a fold, by design).

## k2-0.9b: n = 113 attempts (43 rejected) from 13 runs

| signal | when | AUROC for ok == False [95% CI] | sign flipped (1 - AUROC) |
|---|---|---|---|
| gate self-surprise: mean NLL (n=113) | post-hoc | 0.347 [0.244, 0.459] | 0.653 |
| gate self-surprise: max NLL (n=113) | post-hoc | 0.392 [0.280, 0.507] | 0.608 |
| gate self-surprise: mean_action NLL (n=108) | post-hoc | 0.330 [0.233, 0.436] **(best gate feature, low surprise rejects)** | 0.670 |
| gate self-surprise: mean_think NLL (n=4) | post-hoc | 1.000 [1.000, 1.000] | 0.000 |
| gate scored by 3.7b: mean NLL (n=113) | post-hoc | 0.629 [0.521, 0.730] | 0.371 |
| gate scored by 3.7b: max NLL (n=113) | post-hoc | 0.559 [0.444, 0.670] | 0.441 |
| gate scored by 3.7b: mean_action NLL (n=108) | post-hoc | 0.677 [0.570, 0.782] **(best gate feature, high surprise rejects)** | 0.323 |
| gate scored by 3.7b: mean_think NLL (n=4) | post-hoc | 1.000 [1.000, 1.000] | 0.000 |
| probe, best layer 12 of 29 (LORO-CV, optimistic: best-of-29) | pre-action | 0.802 [0.719, 0.877] | |
| probe, median over layers | pre-action | 0.759 | |
| probe, middle (layer 14) | pre-action | 0.769 [0.680, 0.850] | |
| probe, last block (layer 27) | pre-action | 0.785 [0.698, 0.863] | |
| probe, final norm (layer 28) | pre-action | 0.788 [0.703, 0.867] | |
| probe, mean of last 64 prompt tokens at layer 12 | pre-action | 0.752 [0.663, 0.834] | |
| baseline: prompt length | pre-action | 0.630 [0.524, 0.734] | |
| baseline: step index | pre-action | 0.584 [0.477, 0.689] | |

Probe: standardized features + L2 logistic regression, C=1.0; at the best layer, C=0.01: 0.793, C=0.1: 0.806, C=1.0: 0.802, C=10.0: 0.797.
Per-layer AUROC: 0:0.72 1:0.74 2:0.73 3:0.75 4:0.71 5:0.70 6:0.69 7:0.71 8:0.76 9:0.77 10:0.76 11:0.78 12:0.80 13:0.77 14:0.77 15:0.76 16:0.72 17:0.73 18:0.71 19:0.74 20:0.75 21:0.76 22:0.75 23:0.76 24:0.77 25:0.78 26:0.79 27:0.78 28:0.79.

Spearman between the probe's out-of-fold P(rejected) (best layer) and the gate's surprise, same attempts:

| gate signal | n | Spearman rho [95% CI] | rho on accepted attempts only |
|---|---|---|---|
| self mean NLL | 113 | -0.141 [-0.320, +0.046] | +0.017 |
| self max NLL | 113 | -0.002 [-0.191, +0.192] | +0.171 |
| self mean_action NLL | 108 | +0.045 [-0.149, +0.241] | +0.283 |
| by 3.7b mean NLL | 113 | +0.122 [-0.084, +0.315] | +0.049 |
| by 3.7b max NLL | 113 | +0.178 [-0.021, +0.361] | +0.217 |
| by 3.7b mean_action NLL | 108 | +0.239 [+0.049, +0.427] | +0.230 |

Same-attempt check (joined n=113): gate best feature `mean_action` AUROC 0.330 [0.233, 0.436] vs probe best layer AUROC 0.802 [0.719, 0.877].

Throughput: gate 0.9b: 2342 tok/s (169861 prompt+target tokens, peak 2.0 GB); 3.7b: 596 tok/s (154406 prompt+target tokens, peak 6.8 GB). Probe capture 2454 tok/s (163604 prompt tokens, peak 2.2 GB); probe fit 11 s.

Reasons (joined attempts): ok x65, repeat_call x34, judge_no x7, final x5, unknown_tool:edit_file x1, unknown_tool:edit x1.

## k2-3.7b: n = 50 attempts (11 rejected) from 11 runs

| signal | when | AUROC for ok == False [95% CI] | sign flipped (1 - AUROC) |
|---|---|---|---|
| gate self-surprise: mean NLL (n=50) | post-hoc | 0.284 [0.123, 0.468] **(best gate feature, low surprise rejects)** | 0.716 |
| gate self-surprise: max NLL (n=50) | post-hoc | 0.466 [0.262, 0.673] | 0.534 |
| gate self-surprise: mean_action NLL (n=46) | post-hoc | 0.325 [0.172, 0.507] | 0.675 |
| gate self-surprise: mean_think NLL (n=10) | post-hoc | 0.111 [0.000, 0.375] | 0.889 |
| probe, best layer 1 of 37 (LORO-CV, optimistic: best-of-37) | pre-action | 0.717 [0.516, 0.885] | |
| probe, median over layers | pre-action | 0.568 | |
| probe, middle (layer 18) | pre-action | 0.523 [0.287, 0.758] | |
| probe, last block (layer 35) | pre-action | 0.600 [0.391, 0.796] | |
| probe, final norm (layer 36) | pre-action | 0.528 [0.291, 0.767] | |
| probe, mean of last 64 prompt tokens at layer 1 | pre-action | 0.591 [0.393, 0.777] | |
| baseline: prompt length | pre-action | 0.787 [0.596, 0.946] | |
| baseline: step index | pre-action | 0.580 [0.341, 0.804] | |

Probe: standardized features + L2 logistic regression, C=1.0; at the best layer, C=0.01: 0.661, C=0.1: 0.710, C=1.0: 0.717, C=10.0: 0.754.
Per-layer AUROC: 0:0.70 1:0.72 2:0.63 3:0.57 4:0.57 5:0.57 6:0.51 7:0.53 8:0.53 9:0.54 10:0.54 11:0.51 12:0.58 13:0.58 14:0.65 15:0.63 16:0.58 17:0.53 18:0.52 19:0.49 20:0.55 21:0.48 22:0.48 23:0.47 24:0.52 25:0.54 26:0.58 27:0.58 28:0.58 29:0.60 30:0.60 31:0.62 32:0.61 33:0.61 34:0.65 35:0.60 36:0.53.

Spearman between the probe's out-of-fold P(rejected) (best layer) and the gate's surprise, same attempts:

| gate signal | n | Spearman rho [95% CI] | rho on accepted attempts only |
|---|---|---|---|
| self mean NLL | 50 | -0.466 [-0.648, -0.226] | -0.370 |
| self max NLL | 50 | +0.030 [-0.248, +0.346] | -0.014 |
| self mean_action NLL | 46 | -0.140 [-0.394, +0.138] | -0.109 |

Same-attempt check (joined n=50): gate best feature `mean` AUROC 0.284 [0.123, 0.468] vs probe best layer AUROC 0.717 [0.516, 0.885].

Throughput: gate 3.7b: 578 tok/s (96162 prompt+target tokens, peak 6.6 GB). Probe capture 595 tok/s (89812 prompt tokens, peak 6.5 GB); probe fit 14 s.

Reasons (joined attempts): ok x36, judge_no x5, repeat_call x5, final x3, parse:unclosed_think x1.
