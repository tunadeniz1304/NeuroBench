# Landscape: Keyword FSCIL on MSWC (NeuroBench v1.0) and related work

Literature check done 2026-09-29. Task: 100 base classes, then 10 sessions of 10-way 5-shot (one language
per session, 200 classes by the end). Metrics are base accuracy (session 0) and the average over all 11
sessions, plus footprint, activation sparsity and synaptic ops.
Only numbers found in a source are listed. "unverified" means the number came from a secondary source or
a snippet and not from the primary paper.

## 1. NeuroBench MSWC FSCIL: official entries

| Method | Venue/year | Dataset/task | Reported number(s) | Hardware-plausible incremental rule? | Source URL |
|---|---|---|---|---|---|
| M5 ANN (MFCC + 4 conv blocks, prototype readout) | NeuroBench, Nat. Commun. 16:1545 (2025); leaderboard 2024-01-17 | NeuroBench MSWC FSCIL | Base 97.09% / session avg 89.27%. Footprint 6.03E6 B, act. sparsity 0.783, 2.59E7 dense / 7.85E6 eff. MACs. New-class acc averaged over sessions: 79.61%. Frozen readout (no learning) session avg is 21.41 pts lower, so ~67.86% | partly: the prototype is just the mean of the embeddings, but it is an ANN with float MACs | https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst ; https://www.nature.com/articles/s41467-025-56739-4 |
| SNN (Speech2Spikes, 2 recurrent adLIF layers + linear readout, prototype readout) | same | NeuroBench MSWC FSCIL | Base 93.48% / session avg 75.27%. Footprint 1.36E7 B, 200 Hz exec, act. sparsity 0.916, 3.39E6 dense / 3.65E5 eff. ACs. New-class acc 57.23%. Frozen SNN session avg is 9.97 pts lower, so ~65.30% | partly: the prototype is the mean of spike features written straight into the readout W/b (no backprop), but the float means and biases are not integer or local | same |
| Any newer FSCIL submission | checked 2026-09 | - | **None found.** `leaderboard.rst` on the main, dev, release/2.3.0 and 2025_GC branches lists only the two 2024-01-17 baselines. Newer submissions exist only for other tasks (e.g. tinyRSNN/bigSNN on primate reaching). No open or merged FSCIL-result PRs (FSCIL PRs #1, #116, #149, #150, #158, #171 are all infrastructure). neurobench.ai shows no FSCIL entries | - | https://github.com/NeuroBench/neurobench/pulls?q=is%3Apr+fscil |
| Issue #190 "MSWC pre-training task is very easy" | GitHub, 2024-02 | MSWC FSCIL base set | A 126k-parameter model matches M5 on base; a maintainer reports ~95% test with a GRU of similar MACs (unverified, figures only in a comment) | n/a | https://github.com/NeuroBench/neurobench/issues/190 |

**Key detail (Nat. Commun. text):** the "base accuracy" of 97.09 / 93.48 is the backprop-trained (frozen)
readout. Switching to the prototype readout costs -2.37 pts (ANN) and **-9.17 pts (SNN)** on session 0,
which puts the SNN prototype session 0 at ~84.3%. The paper blames the SNN's weaker session average mainly
on this conversion to a prototype readout, not on forgetting. Per-session curves are only in Fig. 3.

## 2. SNN / low-power FSCIL and continual KWS (2024 onward)

| Method | Venue/year | Dataset/task | Reported number(s) | Hardware-plausible incremental rule? | Source URL |
|---|---|---|---|---|---|
| SAFA-SNN | ICLR 2026 (arXiv 2510.03648) | CIFAR-100, miniImageNet, CIFAR10-DVS, DVS128 Gesture, N-Caltech101 (**no MSWC/KWS**) | miniImageNet last 48.70 / avg 59.45 (vs CLOSER 44.69 / 53.38). DVS128 Gesture 77.91 / 86.74. CIFAR10-DVS 36.96 / 47.56. N-Caltech101 39.69 / 45.68. Energy measured on Jetson Orin | partly: frozen encoder plus prototype updates with orthogonal-subspace projection; float math, GPU-class platform | https://arxiv.org/abs/2510.03648 |
| NC-KWS (ETF classifier) | CCIS vol. 2662, 2026 | FSCIL KWS (dataset not visible in the abstract) | No numbers accessible (paywall) | partly: fixed ETF targets need no classifier learning, but the backbone is aligned by training | https://doi.org/10.1007/978-981-95-5382-2_21 |
| AnalyticKWS | ACL Findings 2025 | Class-incremental KWS on GSC-v1/v2 and SC-100 (**not few-shot, not MSWC**) | ACC 89.48-89.53% on GSC-v2 (T=6/11/21), 87.63-87.99% on SC-100 (T=11/26/51), exemplar-free | partly: closed-form recursive least squares, no backprop, but needs a float matrix inverse | https://arxiv.org/abs/2505.11817 |
| Chameleon (TCN accelerator, 40nm-class ASIC) | IEEE JSSC 61(7), 2026 (arXiv 2505.24852) | Omniglot FSL/CL, GSC-12 | Omniglot 5-way 1-shot 96.8%. 250-class CL on Omniglot, 5-shot: final 79.1%, avg 87.1%. GSC-12 93.3%. Min CL power 12.9 uW, 6.97 uJ per shot | **yes**: on-chip prototype sums converted to FC weights with log2 quantization, multiplier-free (shift + add), 26 B per way. Closest hardware analogue to our goal | https://arxiv.org/abs/2505.24852 |
| Federated FSL on Akida | arXiv 2603.13037 (2026) | GSC v0.02, 3 new words, 2x Raspberry Pi 5 + AKD1000 | 77.0% ± 3.8% (FedUnion, 256-d) vs 63.4% for a single node | yes: Akida on-chip learning with binary weights. FedAvg of binary weights breaks it; neuron concatenation works | https://arxiv.org/abs/2603.13037 |
| Self-learning personalized KWS | IEEE IoT-J 2024 (arXiv 2408.12481) | KWS personalization on MCU | +19.2% / +16.0% over generic models, 8.2 mW labeling power | partly: pseudo-labels from prototype similarity, then on-device fine-tuning (backprop) | https://arxiv.org/abs/2408.12481 |
| Scalable KWS via modular expansion | Interspeech 2026 (arXiv 2607.19918) | Adding new keywords after deployment | New-keyword FRR 6.46 -> 4.37, 16.34M MACs, <=10k added params | no: trains an expansion branch with gradients | https://arxiv.org/abs/2607.19918 |
| ISI-CV gradient-free SNN CL | arXiv 2604.16496 (2026) | Split/Permuted MNIST, Split-N-MNIST | "Zero forgetting" on Split-MNIST/FMNIST (reported qualitatively) | yes (claimed): synaptic importance from inter-spike-interval CV, no gradients | https://arxiv.org/abs/2604.16496 |

## 3. General FSCIL and few-shot KWS methods

Reference numbers are miniImageNet 5-way 5-shot (60 base + 8 sessions), session 0 / last / average, as
reported in the NC-FSCIL (ICLR 2023) Table 1 unless noted.

| Method | Venue/year | Dataset/task | Reported number(s) | Hardware-plausible incremental rule? | Source URL |
|---|---|---|---|---|---|
| ProtoNet | NeurIPS 2017 | FSL | Omniglot FSCIL (from C-FSCIL): 70.61 -> 67.41 | yes: class mean as weights; this is the NeuroBench baseline rule | https://arxiv.org/abs/1703.05175 |
| SimpleShot (centering + L2-norm NCM) | arXiv 2019 | FSL | (not re-checked) | yes: subtract the base mean and normalize, then NCM; cheap and integer-friendly | https://arxiv.org/abs/1911.04623 |
| CEC | CVPR 2021 | miniImageNet / CIFAR-100 | 72.00 / 47.63 / 57.75; CIFAR 73.07 / 49.14 / 59.53 | partly: frozen backbone plus a graph-attention calibration of the classifier | https://arxiv.org/abs/2104.00654 |
| FACT (forward-compatible, virtual prototypes) | CVPR 2022 | miniImageNet | 72.56 / 50.49 / avg 60.79 (TEEN table; the survey also gives 60.79) | partly: base training reserves space; the incremental step is prototypes | https://arxiv.org/abs/2203.06953 |
| C-FSCIL (IBM, hyperdimensional, bipolar prototypes) | CVPR 2022 | miniImageNet, CIFAR-100, Omniglot | Mode 1 (averaged, no gradients): mini 76.37 -> 48.87; Mode 3: 76.40 -> 51.41, avg 61.61. CIFAR Mode 3 77.47 -> 50.47. Omniglot 47-way 5-shot Mode 1 84.16 -> 81.56 | **yes (Mode 1)**: one-pass averaged or bipolarized prototypes in an explicit memory, quasi-orthogonal HD space, built for in-memory computing | https://arxiv.org/abs/2203.16588 |
| ALICE (angular loss, class/data augmentation) | ECCV 2022 | miniImageNet / CIFAR-100 | 80.60 / 55.70 / 63.99; CIFAR 79.00 / 54.10 / 63.21 | partly: training-time feature reservation, NCM at incremental time | https://arxiv.org/abs/2208.00147 |
| LIMIT | TPAMI 2022 | miniImageNet | 72.32 / 49.19 / 59.06 | no: meta-learned transformer calibration | https://arxiv.org/abs/2203.17030 |
| NC-FSCIL (fixed ETF classifier, neural collapse) | ICLR 2023 | miniImageNet / CIFAR-100 / CUB | 84.02 / 58.31 / 67.82; CIFAR 82.52 / 56.11 / 67.50; CUB 80.45 -> 59.44, avg 67.28 | partly: fixed ETF targets are hardware-friendly, but a projection layer is fine-tuned each session | https://arxiv.org/abs/2302.03004 |
| SAVC (supervised contrastive + virtual "fantasy" classes) | CVPR 2023 | CUB200 (mini/CIFAR in supplement) | CUB 81.85 -> 62.50. miniImageNet avg 66.66 (unverified, survey 2502.08181) | partly: contrastive pretraining only; incremental step is frozen prototypes | https://arxiv.org/abs/2304.00426 |
| TEEN (training-free prototype calibration) | NeurIPS 2023 | miniImageNet / CIFAR-100 | mini 73.53 / 52.08 / 61.44 (avg computed from its table); CIFAR 74.92 / 52.64 | **yes**: new prototype = mix of its own mean and a similarity-weighted sum of base prototypes; no training | https://arxiv.org/abs/2312.05229 |
| OrCo (orthogonality + contrast) | CVPR 2024 | miniImageNet (harmonic-mean metric) | aHM 58.12 (vs NC-FSCIL 52.62). Session 0 / 8 HM: 68.71 / 53.12 | partly: pseudo-target reservation at base training; the incremental step still trains | https://arxiv.org/abs/2403.18550 |
| Few-shot KWS in any language (Mazumder) | Interspeech 2021 | MSWC-derived keyword bank, 5-shot | Avg F1 0.75 (180 new words, 9 languages), 0.65 (260 words, 13 unseen languages), 87.4% streaming acc @ 4.3% FAR | partly: small classifier fine-tuned on top of a frozen multilingual embedding | https://arxiv.org/abs/2104.01454 |
| Few-shot KWS with ProtoNets (Parnami & Lee) | ICMLT 2022 | GSC | 94% on 2-way 5-shot (unverified, cited in secondary sources) | yes: prototypes | https://www.researchgate.net/publication/361232070 |
| D-ProtoNets (dummy prototypes, open-set) | Interspeech 2022 | splitGSC FSOSR | "Clear margins" over FSOSR baselines (no numbers checked) | yes: learned dummy/reject prototype at inference | https://arxiv.org/abs/2206.13691 |
| Few-shot open-set on-device KWS (Rusci & Tuytelaars) | arXiv 2306.02161 (2023) | Pretrained on MSWC, evaluated on GSC, 10-shot open-set | Triplet + openNCM: 76% acc @ 5% FAR, AUROC 0.94. ProtoNet 63% / 0.92, DProto 71% / 0.93. DSCNN-S 22k params | yes: openNCM needs no extra parameters; MSWC pretraining is relevant | https://arxiv.org/abs/2306.02161 |

## 4. Hardware on-chip learning references

| Method | Venue/year | Dataset/task | Reported number(s) | Hardware-plausible incremental rule? | Source URL |
|---|---|---|---|---|---|
| BrainChip Akida edge learning | Akida docs (MetaTF) | GSC 32+silence base, +3 new words (4 utterances, x40 augmentation) | 88.80% base -> 88.40% on old classes after learning; 19/19 new-word validation samples correct. Binary weights, 15 neurons/class, num_weights ~1.2x mean spikes | **yes**: last layer only, 1-bit weights and inputs, competitive (k-means-like) multi-neuron-per-class | https://doc.brainchipinc.com/examples/edge/plot_1_edge_learning_kws.html |
| CLP-SNN on Loihi 2 | arXiv 2511.01553 (Intel, 2025) | OpenLORIS few-shot online CL | Matches replay-based accuracy without rehearsal. 0.33 ms / 0.05 mJ per update vs 37.3 ms / 333 mJ on edge GPU | **yes**: self-normalizing three-factor local rule, neurogenesis + metaplasticity, event-driven prototypes | https://arxiv.org/abs/2511.01553 |
| SOEL on Loihi (Stewart, Orchard, Shrestha, Neftci) | IEEE JETCAS 2020 | DVS gesture, online few-shot new classes | Comparable to simulation (no numbers checked) | yes: error-triggered surrogate-gradient update on the last layer, on-chip | https://arxiv.org/abs/2008.01151 |
| e-prop | Nat. Commun. 2020 (Bellec et al.) | RSNN online learning | eligibility trace x learning signal (three-factor) | partly: local traces, but needs a broadcast error signal per neuron | https://www.nature.com/articles/s41467-020-17236-y |
| e-prop on SpiNNaker 2 | Front. Neurosci. 2022 (Rostami et al.) | GSC KWS, trained from scratch on chip | 91.12%, 680 KB, 25k weights, ~12x less energy than a V100 (estimated) | yes: runs on ARM cores in real time | https://www.frontiersin.org/articles/10.3389/fnins.2022.1018006/full |
| ReckOn (Frenkel & Indiveri) | ISSCC 2022 | Navigation, gesture, KWS | 0.45 mm², 28 nm, <150 uW training, 0.8% memory overhead | yes: e-prop-style online RSNN learning on chip | https://arxiv.org/abs/2208.09759 |
| Loihi 2 programmable plasticity | Intel, 2021 (Orchard et al., SiPS) | - | Programmable (microcode) plasticity with third-factor/modulatory traces and graded spikes (feature claims from Intel material, unverified here; this paper covers Loihi 2 signal processing) | yes: target platform for integer three-factor rules | https://arxiv.org/abs/2111.03746 |

## Strongest real competitor on NeuroBench MSWC FSCIL

- **No stronger or newer FSCIL entry exists.** This holds for the NeuroBench leaderboard (all branches, as of
  2026-09), the PRs, neurobench.ai and the literature searched. The only published numbers are still the
  2024 baselines.
- **Strongest overall: M5 ANN, 97.09% base / 89.27% session average** (6.03 MB, 7.85E6 eff. MACs).
- **Strongest SNN: RadLIF RSNN, 93.48% / 75.27%** (13.6 MB, 3.65E5 eff. ACs). Its prototype readout alone
  costs 9.17 pts on session 0.
- Adjacent evidence says the gap can close. On Omniglot, a hardware prototype learner (Chameleon, JSSC 2026)
  reaches 87.1% average over 250 incrementally learned classes with shift/add-only learning. SNN FSCIL work
  (SAFA-SNN, ICLR 2026) does not report on MSWC.
- **Recommended updated target.** The minimum claim is session avg **>= 80%** with an SNN, a clear +5 pts
  over the SNN baseline, which is the SOTA for SNNs on this task. Keep base >= 93.5% with the same readout
  used for the incremental sessions, and make the incremental rule backprop-free and integer.
  - Stretch goal: **>= 85%**, closing more than two thirds of the gap to the M5 ANN.
  - Headline goal: **>= 89.3%**, matching the ANN with fewer ACs and a smaller footprint.
  - Always report the prototype-readout session-0 accuracy. That is where the SNN baseline loses most
    of its points.

## Takeaways for our method

- **Fix the base-to-prototype gap first.** Most of the SNN deficit (-9.17 pts at session 0) comes from
  swapping the trained readout for mean-feature prototypes. Pretrain the SNN so prototypes work well:
  metric/prototypical or supervised-contrastive loss (SAVC), or a fixed ETF/orthogonal target (NC-FSCIL,
  OrCo). Plain cross-entropy is not enough.
- **Use a local incremental rule.** C-FSCIL Mode 1, TEEN and Chameleon all show that a one-pass, no-gradient
  prototype rule is competitive. Use an integer running-sum prototype (count + accumulator) and a log2 or
  bit-shift normalization in place of the division (Chameleon). This maps directly to a Hebbian
  "post = label" update gated by a neuromodulatory "learn/new-class" signal.
- **Calibrate for free.** TEEN-style blending of new prototypes with similarity-weighted base prototypes,
  and SimpleShot centering/L2-norm, add no training and can be done in integer arithmetic. They mainly help
  new-class accuracy, the weak spot (57.23% for the SNN vs 79.61% for the ANN).
- **Plan for binary/bipolar and per-class multi-prototypes.** C-FSCIL shows bipolarized HD prototypes cost
  little accuracy. Akida uses binary weights with several neurons per class plus competition. Both suggest
  1-2 bit readout weights with k prototypes per class are viable and cut footprint.
- **Reserve feature space at base time** with virtual classes or orthogonal targets (FACT, ALICE, SAVC, OrCo).
  Without it, the ~100 new multilingual words crowd into the base clusters, and the base task is already
  "too easy" (issue #190), so capacity is available.
- **A three-factor, neuromodulated rule on Loihi 2 is proven for exactly this pattern:** local Hebbian
  plus a modulatory gate, self-normalizing weights and neurogenesis for new classes (CLP-SNN: 0.05 mJ per
  update). Frame our rule the same way and report synaptic-op and energy cost per incremental update, not
  only inference cost.
- **Report everything NeuroBench asks for, plus per-session and new-class-only accuracy.** The ANN wins on
  accuracy but needs ~21x more effective ops. An SNN at >= 80-85% with ~1E5-3E5 ACs and a smaller footprint
  would be a clear Pareto improvement.
