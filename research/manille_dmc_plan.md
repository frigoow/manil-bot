# Manille Bot — DMC Blueprint Plan (with Gated Transformer Upgrade)

## 0. Purpose and context

This plan replaces the previous ExIt/distillation approach, which plateaued at teacher (ISMCTS) level and is abandoned. The new paradigm follows the DouZero/Hanabi lineage: **learn a blueprint policy from scratch with Deep Monte Carlo (DMC) self-play — no search teacher anywhere in training** — so that card inference and partner conventions can emerge in the policy itself. Test-time search (SPARTA-style) is a later phase built on top of the frozen blueprint; this document specifies the blueprint.

**Environment and constraints (unchanged from before):**
- Simplified Manille variant: no bidding, no meegaan/doubling; trump selection is **rule-based** (fixed heuristic, outside all learning; identical selector in training and evaluation). Objective: **maximize expected team card points per deal** (linear, zero-sum, 60 total; manille=5, A=4, K=3, Q=2, J=1; rank order 10 > A > K > Q > J > 9 > 8 > 7).
- Existing assets: authoritative rules engine (C++ and Python), performant C++ ISMCTS (now **evaluation opponent only**, never a teacher), hard-constraint consistent-deal sampler. PIMC is a dead end — do not use.
- Hardware: single local RTX 3060 (12 GB) + multicore CPU. Runs are multi-day and unattended: everything checkpoints/resumes, all gates are automated, and progress is written to a machine- and human-readable log.
- Standing benchmarks for every checkpoint (duplicate-deal harness, teams swapped, bootstrap CIs, ≥2,000 deals): (a) **vs uniform-random-legal agents** — calibration anchor: **≥ 37 points/deal is very strong for a pure NN**; (b) **vs ISMCTS at 5,000 iterations** — beating any ISMCTS with a search-free NN is a major, previously unachieved milestone; report prominently if it occurs. Also track vs ISMCTS-100 and ISMCTS-1000 for a finer strength ladder.

**Why DMC here (summary of the argument):** 8 decisions per player per episode, dense linear points reward, near-free CPU simulation → full-episode Monte Carlo returns are low-variance and unbiased, and none of TD's machinery (bootstrapping, target networks, max-operator bias) is needed. Prior PPO and ExIt failures are consistent with exactly the instabilities DMC removes. Joint training of partners on shared team return allows conventions to emerge — the thing the search-teacher pipeline structurally prevented.

---

## 1. Core algorithm: Deep Monte Carlo with action-as-input Q

### 1.1 Q-function formulation

Train **Q_θ(s, a) → scalar**: expected final team points (normalized: (points − 30)/30) given information state s and candidate action a.

- **Action-as-input**: the candidate card is encoded as input features (card identity, suit-relative-to-trump, rank, point value, trick-winning status if played now, etc.), and the network outputs one scalar. To act, batch-evaluate all legal actions (≤ 8) and take argmax (or epsilon-greedy during training). This generalizes across cards and needs no output masking.
- **One shared network for all four seats**, with all features encoded seat-relative (own hand, partner = seat+2, opponents = seats+1/+3). Team symmetry and seat rotation are free data augmentation.
- **Return target**: the final normalized team points of the episode, attached to every decision the player made in it. No bootstrapping, no target network, no TD(λ). Points already banked during the episode appear in the input features, so the network learns the remaining-value structure implicitly.

### 1.2 Self-play data generation

- All four seats play epsilon-greedy with the current θ (partners are always the current policy — cooperative consistency).
- **Epsilon ladder across actors** (Ape-X style): N parallel actor processes, actor i uses ε_i from a fixed spread, e.g. ε ∈ {0.30, 0.20, 0.12, 0.07, 0.04, 0.02, 0.01, 0.005}. Evaluation always uses ε = 0.
- **Opponent league**: with probability ~20%, the *opposing team* is a frozen past checkpoint sampled from the last ~10 saved checkpoints (uniform or prioritized by recency). The partner is never a frozen checkpoint. This damps best-response cycling in the zero-sum team setting.
- Per-episode record (for every decision point): encoded features, chosen action, ε used, final team points, **true hidden hands** (for the belief net and diagnostics), checkpoint version.

### 1.3 Training loop

- **Alternate generate/train epochs** rather than fully concurrent actor-learner (avoids GPU contention on the 3060; with 32-ply episodes this loses little). Cycle: generate G deals on CPU actors (NN inference batched on GPU, FP16) → train T gradient steps → checkpoint + benchmark → repeat.
- Loss: MSE (or Huber δ=1) between Q_θ(s, a) and the normalized return.
- Buffer: sliding window of the freshest ~2–5M decision points (tune so a window is consumed in ~4–8 passes). DMC works on-policy-ish; do not hoard stale data.
- Optimizer AdamW, lr 1e-4 → 1e-5 cosine (DMC is regression — it tolerates and benefits from a somewhat lower lr than policy-gradient setups), batch 2048, grad clip 1.0, EMA weights for evaluation and checkpointing.
- Suggested starting scale: G ≈ 50k deals per cycle, T sized to ~2 passes over new data. Tune from measured throughput, not from these numbers.

### 1.4 Belief network (unchanged purpose, same data stream)

Train B_φ alongside (separate net or a second head on a shared trunk): predict the three hidden hands from the information state. **All design rules from the previous plan's §5 carry over verbatim and remain mandatory**: per-unseen-card 3-way softmax over hidden seats parameterized as logit offsets on the constrained-uniform prior; hard constraints applied structurally (seen-card exclusion, void masking, hand-size renormalization); per-play legal-set summaries in the input; no label smoothing; per-trick Δlog-loss evaluation vs constrained-uniform; memorization and probe-task diagnostics before trusting it. B_φ is trained only on fresh data from the current policy generation. It is not needed by the blueprint to act (beliefs are implicit in Q_θ), but it is (a) the convention-expression diagnostic instrument (§3) and (b) the belief filter for SPARTA later (§5). Train it from day one; it is cheap.

---

## 2. Architecture A (baseline): feedforward over engineered history

Start here. This must be fully working and benchmarked before any transformer work.

**Input features (seat-relative, fixed-size vector; golden-tested identically in C++ and Python):**
- Own hand: 32 binary.
- Per-seat played cards: 4 × 32 binary (who played what — the inference substrate).
- Derived void flags: 4 seats × 4 suits (engine-computed, exact).
- Current trick: led suit, cards on table with relative-seat and play-order, current winning seat and card, points on the table, own position in trick (1st–4th).
- Per-trick summary for each completed trick t = 1..7: leader, winner, points captured, and for each seat a compact record (card played + a **legal-context summary at that moment**: could-follow / could-trump / could-overtrump / number of legal moves). The legal-context bits are what make "chose X while Y was available" visible — the entire inference and convention signal lives there; do not omit them.
- Trump suit (4), seat that "owns" trump per the rule-based selector if the rules make that public.
- Scalars: trick number, own-team and opponent-team points so far (normalized), cards remaining.
- Candidate action features (§1.1).

**Network:** MLP trunk, 4–6 residual blocks × 512–1024 units, GELU, layernorm; Q head = 2-layer MLP → scalar. Optional cheap auxiliary head: predict partner's void pattern (adds inference pressure early). Keep total < 5M params, FP16 inference.

**Acceptance for Architecture A (gates, automated):**
- A-1: Learning-curve sanity — beats random benchmark by > 31 pts/deal within the first benchmark cycles and improves monotonically (smoothed).
- A-2: ≥ 37 pts/deal vs random (the "very strong NN" anchor).
- A-3: Beats ISMCTS-100; report the full ladder (100/1,000/5,000). A win vs ISMCTS-5000 is the headline stretch result.
- A-4: B_φ trained on this stream beats constrained-uniform per-trick log-loss with the expected growing-gap-by-trick shape.

---

## 3. Convention-expression diagnostics (the gate for the transformer upgrade)

The transformer is **not** an automatic upgrade. It is triggered only by evidence that the policy is informative-play-limited in ways the engineered history cannot express. Run this diagnostic suite at every benchmark checkpoint; it is cheap and fully automatable.

**D1 — Signaling information curve.** Using B_φ (or a dedicated probe net), measure per-trick log-loss of predicting a player's hand *from that player's own actions* (i.e., how informative is each seat's play about its holdings, from the partner's viewpoint). Track this curve across training generations. Healthy convention development = the tricks-2–4 information gap grows over generations. A plateau in this curve *while strength also plateaus* is the primary under-expression signal.

**D2 — Convention-reliance (intra-lineage cross-play).** Pair the current policy with its own checkpoint from ~N generations ago as partner (opponents = current). Self-play score minus this cross-play score measures how much strength depends on recently co-adapted conventions. Near-zero forever = conventions aren't forming (or aren't needed); large and growing = they are forming and load-bearing.

**D3 — Counterfactual signal response.** Sample positions where the actor's top-2 Q values are within a small margin (genuinely free choices). Test (a) whether the choice among near-ties correlates with private holdings (signal sent), and (b) whether, in matched continuations, the partner's subsequent Q-ranking responds to which near-tie was chosen (signal received). Both measurable offline from logged data + forward passes.

**D4 — History-representation saturation (the decisive test for the upgrade).** Train a small probe on frozen features: predict the true hidden hands from (i) the engineered feature vector vs (ii) the raw ordered ply sequence (a tiny sequence model probe). If (ii) beats (i) by a meaningful margin, information exists in the raw history that the engineered encoding destroys — direct evidence the *representation* is the bottleneck, not the algorithm.

**Upgrade trigger (all three):** strength plateaued across ≥ 3 consecutive benchmark cycles (CI-overlapping) **and** D1 flat **and** D4 shows a raw-sequence advantage. If strength plateaus but D4 is null, the bottleneck is elsewhere (scale, ε schedule, league) — do not reach for the transformer; log the finding and continue on the tuning axes instead.

---

## 4. Architecture B (gated upgrade): small transformer history encoder

Same DMC algorithm, same targets, same buffer, same benchmarks — **only the encoder changes**. The Q head interface is identical so the swap is drop-in and A/B-comparable.

**Tokenization:**
- One token per historical play (≤ 31): embedding = card (32) + relative seat (4) + trick index (8) + position-in-trick (4) + legal-context summary at that moment (the could-follow/could-trump/count features) + led-suit and winning-so-far context. Sum or concat+project to d_model.
- Plus: 8 tokens for own current hand cards (flagged as hand tokens), one token for the current-trick/global state (trump, scores, trick number), and one token for the **candidate action**.
- Output: final hidden state of the candidate-action token → Q head (action-as-query pattern; keeps ≤ 8-action batching cheap because history tokens can be encoded once per decision and reused across candidate actions via KV caching if profiling justifies it — implement the simple recompute version first).

**Model:** pre-LN transformer, d_model 128, 4 layers, 4 heads, FFN 4×, learned positions, ~1.5–2.5M params, FP16. Sequence length ≤ ~42 tokens — this is tiny; inference cost will be dominated by batching efficiency, not FLOPs.

**Comparison protocol (both required, reported side by side):**
- Matched **games-seen**: train B on the same cumulative deal count as A had at its plateau.
- Matched **wall-clock** on the 3060 (B is slower per step; this is the honest deployment comparison).
- B is adopted only if it beats A on the standing benchmarks in the matched-wall-clock comparison **and** D1/D4 gaps close (i.e., it actually expresses what was missing — otherwise the win came from somewhere else and the diagnosis was wrong; log that).

---

## 5. Follow-on phases (out of scope here, design for them now)

- **Endgame hybrid (cheap, early win):** at trick T ≥ 6, replace the NN action choice with exact search over the belief-filtered set of consistent deals (enumerate or sample deals consistent with history and weighted by B_φ; solve the ≤ 3-trick remainder exactly with all players modeled as the blueprint or as maximizers). This patches the NN's structural endgame leak against ISMCTS-5000. Keep it behind a flag so pure-NN benchmarks stay clean.
- **SPARTA-style test-time search** on the frozen blueprint: belief filtering by "would the blueprint have played what I observed" (B_φ as proposal, blueprint likelihoods as weights), single-agent lookahead with everyone else following the blueprint. The blueprint + B_φ from this plan are exactly its required inputs — nothing built here is throwaway.
- If self-play conventions ever become a legibility problem, OBL returns as a *regularizer on the DMC loop* (fictitious-transition targets), not as a teacher.

---

## 6. Autonomous-run guardrails

- Every cycle writes: checkpoint, benchmark table (random / ISMCTS-100 / 1,000 / 5,000 with CIs), D1–D4 diagnostic values, throughput stats, to an append-only run log. Milestones (37+ vs random; first win vs each ISMCTS tier) are flagged loudly in the log.
- Hard stop conditions (halt and summarize rather than burn days): NaN/inf in loss; benchmark regression > X points sustained over 3 cycles; buffer starvation; GPU OOM loops.
- Never report a gate as passed without the harness numbers attached. All strength claims come from the duplicate-deal harness, nothing else.
- Resume-from-checkpoint must be exercised in the validation run (kill and restart mid-cycle once, verify bitwise-identical config and continuing curves).

## 7. Milestones

| # | Milestone | Gate |
|---|---|---|
| M0 | Actors + buffer + benchmarks + feature golden tests + resume test | Throughput measured; ISMCTS ladder ordering sanity check; kill/resume verified |
| M1 | **Validation run**: Architecture A, small scale (~200–500k deals) | Learning curve clearly above random-agent floor and rising; B_φ beats uniform; full pipeline exercised end-to-end (~1–2 days) |
| M2 | Architecture A main run to plateau | A-1..A-4 gates; full ISMCTS ladder reported; D1–D4 logged every cycle |
| M3 | Upgrade decision | Trigger conditions of §3 evaluated and the decision + evidence logged (either outcome is a valid result) |
| M4 | (If triggered) Architecture B trained and compared | §4 comparison protocol, both matched-games and matched-wall-clock tables |
| M5 | Frozen blueprint + B_φ handoff for endgame-hybrid / SPARTA phase | Final benchmark table; artifacts: weights, feature spec version, run log |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Best-response cycling between team lineages | Opponent league of frozen checkpoints; monitor benchmark-vs-fixed-opponents (not just self-play loss) for progress |
| Conventions emerge slowly under ε-greedy (signal only pays if partner already responds) | Volume is the primary answer (episodes are cheap); D1/D3 detect emergence; auxiliary void-prediction head adds inference pressure; ε ladder keeps some high-exploration actors alive throughout |
| Plateau misdiagnosed as representation limit | §3 trigger requires D4 evidence, not just a flat curve; alternative axes (scale, lr, ε, league mix) enumerated as the default response |
| Transformer wins for the wrong reason (capacity, not sequence info) | Adoption requires D1/D4 gaps to close, not just a benchmark win |
| Q overestimation on rarely-taken actions poisons argmax | ε ladder guarantees coverage of all legal actions; monitor per-action visit counts in logs; Huber loss |
| 3060 throughput bottleneck | FP16, batched inference, alternate generate/train; feature encoding in C++; transformer gated so the cheap architecture runs first |
