# FedSynAI — Review-1 prototype (≈30 %)

Privacy-preserving collaborative AI: Federated Learning + CTGAN synthetic tabular data + differential privacy + membership-inference evaluation.
Dataset: UCI Adult income, split across 4 simulated clients (IID or Dirichlet non-IID).

## Run
    pip install -r requirements.txt
    python tests_smoke.py                 # ~20 s sanity check
    streamlit run app.py                  # dashboard + live FL demo (uses stored results/results.json)
    python scripts/run_experiments.py     # regenerate all results (~5 min on CPU)

## Layout
| File | Purpose |
|---|---|
| fedsynai/data.py | load/preprocess Adult, IID & Dirichlet non-IID partitioner |
| fedsynai/fl.py | FedAvg, FedProx, communication accounting, centralized baseline |
| fedsynai/privacy.py | update clipping + Gaussian noise (client-level DP), RDP ε accountant |
| fedsynai/synth.py | per-client CTGAN, fidelity (KS/TVD), distance-to-closest-record |
| fedsynai/attacks.py | loss-threshold membership inference |
| fedsynai/experiments.py | E1 central · E2 FedAvg IID · E3 non-IID (FedAvg/FedProx) · E4 FL+DP · E5-lite FedSynAI |

## Results (seed 42, 20 rounds)
| Exp | Acc | F1 | AUC |
|---|---|---|---|
| E1 centralized | 0.849 | 0.654 | 0.910 |
| E2 FedAvg IID | 0.851 | 0.680 | 0.910 |
| E3a FedAvg non-IID | 0.836 | 0.549 | 0.908 |
| E3b FedProx non-IID | 0.837 | 0.557 | 0.908 |
| E5-lite (synthetic augmentation) | 0.842 | 0.599 | 0.902 |

## Honest limitations (say these in the review)
- DP is client-level with only 4 clients; useful utility only at ε in the hundreds. Not yet a meaningful guarantee.
- E5-lite shares synthetic rows from non-DP CTGANs and uses the global class rate as a public statistic; it recovers part, not all, of the non-IID loss.
- Membership inference is the simple loss-threshold attack (AUC ≈ 0.51). Stronger attacks, secure aggregation, Opacus, Flower, MLflow, CTAB-GAN+ and extra datasets are Review-2 work.
