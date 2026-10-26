# Learn-then-Test (LTT) Correctness Audit for VEIL-X Decisions

> Phase 1a, 2026-10-26. Implementation: `src/veilx/selective/ltt.py`. Simulation checks:
> `reports/simulations/*.json` (SIMULATED; registry SIM-002…SIM-007).
> Source: Angelopoulos, Bates, Candès, Jordan, Lei, *Learn then Test*,
> [arXiv:2110.01052](https://arxiv.org/abs/2110.01052). This audit read §1–3 via the
> ar5iv rendering; equation numbers are re-checked against the PDF before citation
> (evidence PUB-001).

## 1. Decision problem

A pair x = (image A, image B) with label y ∈ {1 genuine, 0 impostor}, a fixed score s(x), and
a covariate c(x) (e.g. co-visibility). The candidate decision rules are λ = (τ_low, τ_high, c_min):

T_λ(x) = MATCH if s ≥ τ_high (and c ≥ c_min); NON-MATCH if s ≤ τ_low (and c ≥ c_min);
otherwise ABSTAIN.

**Accepted** means T_λ(x) ≠ ABSTAIN. Λ = (λ₁, …, λ_J) is a finite, pre-specified list.

## 2. Estimands

| Name | Definition | Depends on prevalence? |
|---|---|---|
| E(λ), pooled accepted error | P(T_λ(X) ≠ Y \| T_λ(X) accepted) | **Yes** |
| sFMR(λ) | P(T_λ = MATCH \| accepted, Y = 0) | No |
| sFNMR(λ) | P(T_λ = NON-MATCH \| accepted, Y = 1) | No |

**Lemma (prevalence robustness).** For any prevalence π,
E_π(λ) = [π·a₁·sFNMR + (1−π)·a₀·sFMR] / [π·a₁ + (1−π)·a₀], where a_y = P(accepted | Y = y).
This is a convex combination of sFNMR and sFMR, so E_π(λ) ≤ max(sFMR, sFNMR).
*Consequence:* certifying sFMR ≤ α_FM and sFNMR ≤ α_FNM certifies E_π ≤ max(α_FM, α_FNM)
for **every** π. **Decision D-019:** the class-conditional pair is the **primary** estimand,
and E(λ) at the calibration prevalence is secondary.

## 3. Is the binomial test valid for a *conditional* risk?

The PI asked specifically whether a binomial test validly controls
P(prediction incorrect | prediction accepted).

**Claim.** Assume:
- calibration pairs Z₁…Z_n are i.i.d.;
- for a fixed λ, the acceptance indicator Aᵢ and the error indicator Eᵢ are functions of Zᵢ
  alone (assumption A7: *acceptance locality*);
- m = Σ Aᵢ and k = Σ AᵢEᵢ.

Then, conditional on {Aᵢ}, the accepted Zᵢ are i.i.d. from the law of Z given acceptance, so
k | m ~ Binomial(m, E(λ)). Hence p(λ) = P(Bin(m, α) ≤ k) satisfies
P(p(λ) ≤ u | m) ≤ u for every u whenever E(λ) > α (binomial monotonicity in the success
probability). Averaging over m keeps it super-uniform, and p := 1 when m = 0.

*This is exactly the construction LTT itself uses for selective classification (§3.2: exact
binomial tail on the n(T) accepted points).*

- **Class-conditional version.** The same argument holds within each class: accepted
  impostors give a binomial count of false matches, and accepted genuines a binomial count
  of false non-matches. The null is the union {sFMR > α_FM} ∪ {sFNMR > α_FNM}. The
  intersection–union p-value max(p_FM, p_FNM) is valid with no δ split.
- **Multiple λ.** LTT's Theorem 1 requires only per-hypothesis valid p-values and a FWER
  procedure. We use either:
  - fixed-sequence testing, with the order pre-specified and testing stopped at the first
    non-rejection (valid under arbitrary dependence among the p-values), or
  - Bonferroni.

  Choosing any rule in the rejected set Λ̂ (we take maximal calibration coverage) keeps the
  guarantee, because Theorem 1 bounds sup_{λ∈Λ̂} R(λ).

**Conclusion: the naive binomial test is valid for the conditional risk under the conditions
below.** It becomes **invalid** when any of them fails. The implementation checks every
condition it can check mechanically.

| # | Condition | How it can fail in VEIL-X | Enforcement |
|---|---|---|---|
| A1 | Cal and test pairs are exchangeable | Held-out families, ROF, other benchmarks, prevalence change (for E) | Guarantee is never claimed outside Test-ID; the scope string says so |
| A2 | Cal pairs are independent | Shared identities, images, source photos, occlusion variants, duplicate files | `PairSet.summary.identity_disjoint` and one use per root-source photo (content-hash keyed) are required, else `ProtocolViolationError` |
| A7 | Acceptance locality | A threshold such as τ₀, a margin, strata edges, a temperature or the candidate order fitted on **cal** | All rule parameters, strata edges and the candidate order come from train/val (`order_candidates_on_validation`). Cal data are only *counted* |
| A6 | Pre-specified Λ and order | Grid or order chosen after looking at cal | Order is computed on validation data only; documented |
| Mixture E only | Pairs i.i.d. from the π-mixture | Fixed genuine/impostor counts (the stratified design) make k a sum of two binomials, not Binomial(m, E) | Mixture estimand refused unless `class_assignment = "bernoulli"` |
| Stratified | Strata defined per pair, by rules fixed in advance | Strata cut at calibration quantiles | Strata edges must be fixed on validation; per-stratum δ/S (Bonferroni over strata) gives simultaneous control, hence overall control (convex combination) |

**What is *not* controlled:**
1. Risk under distribution shift (held-out families, real occlusions).
2. Risk conditional on an identity, stratum or family, except per-stratum under stratified
   calibration.
3. The *realised* error on a finite test set. The guarantee concerns the population risk;
   finite test sets fluctuate around it.
4. Any property when the calibration set violates A2 or A7. `require_valid=False` then yields
   `guarantee_valid=False` and the scope "NO statistical guarantee".

## 4. A pitfall found and fixed: fixed-sequence power collapse

The first implementation ordered rules from widest to narrowest abstention band. The widest
bands accept almost nothing, so p = 1 and fixed-sequence testing stopped at step 1, which
made the calibrator infeasible in every run of the first quick simulations. That is valid but
useless. The fix is `order_candidates_on_validation`, which does two things on **validation**
data:
- prunes rules whose expected accepted count per class (scaled to the calibration size)
  cannot reach the zero-error minimum ⌈log δ / log(1−α)⌉;
- orders the rest by increasing validation risk.

Because the order never depends on calibration data, validity is unaffected.

## 5. Simulation checks (SIMULATED, synthetic score model; not face-recognition evidence)

| Check | Result (registry id) |
|---|---|
| Violation rate with independent calibration ≤ δ | 0.5–1.0% at δ = 5% (SIM-003); 2.5% at δ = 10% (SIM-002); a unit test asserts it is ≤ δ + 3 MC s.e. |
| Dependent (identity-reusing) calibration | Violations rose to 3.75% (120 identities) and 8.25% (30 identities) at δ = 10%. Below δ here, but not guaranteed (SIM-002) |
| Distribution shift | 0.5% → 16% → 95.5% → 100% violations as synthetic shift severity grows (SIM-004) |
| Prevalence | Mixture estimand: 9% violations at test prevalence 0.95 (δ = 5%). Class-conditional: ≤ 0.5% at every prevalence (SIM-007) |
| Stratified vs global | Covariate-only shift: stratified 1% vs global 100% violations, at 17% vs 87% coverage. Conditional shift: both fail (SIM-005) |

## 6. Empirical-only alternative

`EmpiricalCalibrator` picks the highest-coverage rule whose *empirical* calibration risk meets
the target. It is always labelled `guarantee_valid=False`, and it is the only acceptable
option whenever A2 or A7 cannot be met.

## 7. Open items
- Re-check §3.2 equation numbering against the arXiv PDF (PUB-001 caveat).
- The Hoeffding–Bentkus p-value is not implemented. For {0,1} losses the exact binomial is
  the natural, sharper choice (LTT §3.2).
