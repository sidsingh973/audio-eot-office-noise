# Office-noise end-of-turn evaluation

500 English clips from smart-turn-data-v3.2-test. Office-5 noise is unseen by every model. Each model gets its deployment input (C: noise-suppressed; others: raw). Thresholds from each model's own val set at a 5% cut-in budget.

## Headline: office-5 at 10 dB

| Model | PR-AUC [95% CI] | ROC-AUC | Recall @ ≤5% cut-in (best thr) | Val thr → recall | Val thr → cut-in |
|---|---:|---:|---:|---:|---:|
| Smart Turn v3.2 (fp32) | 0.981 [0.971, 0.990] | 0.982 | 0.913 | 0.942 | 0.074 |

## PR-AUC by noise level

| Model | native | office-5 20 dB | 10 dB | 5 dB | 0 dB* | office-3 10 dB (seen) |
|---|---:|---:|---:|---:|---:|---:|
| Smart Turn v3.2 (fp32) | 0.990 | 0.986 | 0.981 | 0.962 | 0.929 | 0.976 |

## Recall at ≤5% cut-in by noise level

| Model | native | office-5 20 dB | 10 dB | 5 dB | 0 dB* | office-3 10 dB (seen) |
|---|---:|---:|---:|---:|---:|---:|
| Smart Turn v3.2 (fp32) | 0.959 | 0.942 | 0.913 | 0.868 | 0.661 | 0.872 |

## Cut-in rate at the val threshold (drift under noise)

| Model | native | office-5 20 dB | 10 dB | 5 dB | 0 dB* | office-3 10 dB (seen) |
|---|---:|---:|---:|---:|---:|---:|
| Smart Turn v3.2 (fp32) | 0.058 | 0.062 | 0.074 | 0.112 | 0.225 | 0.093 |

*0 dB is louder than anything in training (5-20 dB).

## Full matrix: PR-AUC for every model on every input

| Model | native raw / NS | office-5 20 dB raw / NS | 10 dB raw / NS | 5 dB raw / NS | 0 dB* raw / NS | office-3 10 dB (seen) raw / NS |
|---|---:|---:|---:|---:|---:|---:|
| Smart Turn v3.2 (fp32) | 0.990 / 0.991 | 0.986 / 0.988 | 0.981 / 0.980 | 0.962 / 0.965 | 0.929 / 0.915 | 0.976 / 0.973 |

## Native test split by the clip's own background level (Pipecat already mixed some noise in)

| Model | quiet (< -48 dBFS) | middle | noisy (> -41 dBFS) |
|---|---:|---:|---:|
| Smart Turn v3.2 (fp32) | 0.998 | 0.989 | 0.982 |
