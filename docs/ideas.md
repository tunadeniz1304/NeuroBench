# Candidate ideas (Phase 4)

Scores 1–5: **Gain** = expected session-average gain on the pseudo protocol; **Cost** = implementation +
compute cost on one T4 (5 = cheap); **HW** = hardware plausibility of what runs at deployment /
in incremental sessions (5 = trivially integer, local, event-driven). Priority = Gain + Cost + HW.

Observation driving most ideas: the baseline readout `w_c = 2 m_c, b_c = −|m_c|²/T` summed over time
is exactly nearest-class-mean with squared Euclidean distance on raw spike-count vectors. The backbone was
trained for a linear softmax readout, not for NCM; novel-class accuracy is only ~57% (paper).

| # | Idea | Where | Gain | Cost | HW | Prio | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Centered + L2-normalized integer prototypes** (SimpleShot CL2N on spike counts). Score `S·ŵ_c + b_c`, `ŵ_c = q((P_c−μ)/‖P_c−μ‖)`, `b_c = −μ·ŵ_c`. Sample norm drops out of argmax. | B | 4 | 5 | 5 | 14 | μ = base mean spike count (1024 ints) stored once; per-class norm computed once at learning time |
| 2 | **Three-factor Hebbian prototype rule**: `Δw_ci = pre_i(t) · post_c(t) · M(t)`, post = label/teacher spike, M = novelty gate (on during a learning session). Integer accumulators, then shift-normalize + stochastic rounding to k-bit weights. Mathematically = idea 1's prototype. | B | 0 (enabler) | 5 | 5 | 10 | The core contribution; bit-exact `hw_model/` reference |
| 3 | **Cosine-classifier + temperature pretraining** (normalized spike counts vs normalized class weights) so that training matches the deployment NCM readout. | A | 4 | 4 | 5 | 13 | Backprop only offline |
| 4 | **Spike-train augmentation**: time/channel masking (SpecAugment on S2S), time shift, spike dropout/jitter, mixup. | A | 3 | 5 | 5 | 13 | Generalization to unseen languages |
| 5 | **Supervised contrastive / prototype-margin loss** (SupCon or ArcFace-style margin) on spike-count embeddings. | A | 3 | 3 | 5 | 11 | Tighter clusters → better 5-shot prototypes |
| 6 | **Forward-compatible virtual classes** (FACT-style: mixup of two classes as a new virtual class; reserve logits) | A | 2 | 3 | 5 | 10 | Reserve embedding space for future classes |
| 7 | **Deterministic zero initial state** instead of upstream random membrane init | A/HW | 1 | 5 | 5 | 11 | Removes stochastic inference; needed for bit-exact HW |
| 8 | **Novel-prototype calibration (TEEN-style)**: shift few-shot prototypes toward similar base prototypes, weights from readout activations of the novel prototype | B | 2 | 4 | 3 | 9 | Needs reading other classes' weights → only partly local |
| 9 | **Scalar novel-class bias calibration** from local statistics (per-class norm / count-based shrinkage) | B | 1 | 5 | 4 | 10 | Counter base–novel imbalance (500 vs 5 samples) |
| 10 | **Smaller / sparser backbone**: hidden 512–768, firing-rate regularization, fewer Eff_ACs | C | −1..0 | 4 | 5 | 9 | Needed for win condition 2 |
| 11 | **Low-bit weights**: QAT to 8/4-bit for W, V; 4–8 bit prototypes | C | −0.5 | 3 | 5 | 8.5 | Footprint |
| 12 | **Adaptive thresholds / learnable delays** in the backbone | A | 1 | 2 | 3 | 6 | Expensive on T4 |
| 13 | **Multi-timescale readout features** (counts from both hidden layers) | A/B | 1 | 4 | 4 | 9 | Richer embedding at no training cost |
| 14 | **Base weights kept as trained normalized classifier vs replaced by prototypes** | B | 1 | 5 | 5 | 11 | Decide on pseudo protocol |

## Selected for Phase 4 (top by priority, cheapest first)
1. Idea 1 + 2 + 7 — integer CL2N prototypes learned by a three-factor rule (incremental learner).
2. Idea 3 + 14 — cosine-classifier pretraining aligned with the deployment readout.
3. Idea 4 (+5 if 3 saturates) — augmentation / contrastive representation.
4. Idea 10 + 11 — efficiency to meet the ops/footprint win condition.

Each is evaluated only on the pseudo protocol (fold 0; decisions confirmed on fold 1).
Dropped after 2 non-helping attempts; changes within ±0.5 pts are treated as noise.
