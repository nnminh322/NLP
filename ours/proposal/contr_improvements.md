# GSR-CACL Architecture Improvements: P0 & P2

> **Version note:** Both improvements are **opt-in** via CLI flags. The original behavior is preserved as `v1` — no breaking changes.
> ```bash
> # Use v1 (original): --contr1 v1 --contr2 v1
> # Use v2 (improved): --contr1 v2 --contr2 v2
> ```

---

## P0 — Scale-Aware Numeric Encoding

### The Problem

Financial tables contain numbers at vastly different scales:

```
Apple Inc. (Fortune 500):
  Revenue         $ 298,500,000,000   (≈ 3×10¹¹)
  R&D Expenses    $    100,000,000    (≈ 1×10⁸)
  Gross Margin              46.3%     (≈ 4.6×10⁻¹)
  EPS                   $ 6.57
  Shares Outstanding  15,000,000,000 (≈ 1.5×10¹⁰)
```

These range from **billions to fractions of a percent** — six orders of magnitude apart.

The original numeric encoder represents each number as:

```python
features = [log₁₊₁(|v|), sign, is_zero, bucket]
# → [26.2, 1.0, 0.0, 4.0] for $298B revenue
# → [18.4, 1.0, 0.0, 3.0] for $100M R&D
```

The problem: `$298B` and `$100M` both look like "large numbers" to the GAT. Their difference is only **7.8 log units** despite a **3,000×** difference in real value. The encoder collapses magnitude distinctions that matter enormously in financial context.

A `$1B revenue` vs `$500M revenue` question requires the model to understand: these are from **different revenue tiers**, not just different numbers. The original encoder cannot make this distinction.

### Why the Old Method Was Reasonable

The original log-scale approach (`log₁₊₁(|v|)`) is a well-established technique:

- **Logarithm** naturally compresses scale: `log(10ⁿ) = n`, so billion→11, million→6
- **Sign** captures direction (profit vs loss)
- **Bucket** groups by rough magnitude (K/M/B)
- It's **lightweight**: only 4 features, no learnable embeddings

For general NLP or non-financial tables, this works fine. The issue only appears when:
1. Tables mix multiple magnitude tiers (revenue tier + margin tier + share-count tier)
2. The GAT needs to distinguish cells at different scales

### The Improvement

**Idea:** Separate *order of magnitude* from *coefficient*, and add *unit awareness*.

```
A number like "$3.14 billion" has three distinct parts:
  • Order of magnitude:  10⁹  (billion)
  • Coefficient:         3.14
  • Unit:                dollar / percent / shares
```

Instead of one log feature, use three:

| Component | What it encodes | Example for `$298.5B` |
|---|---|---|
| **Magnitude bin** | Which power of 10 does \|v\| fall in? | Bin 11 (= 10¹¹) |
| **Mantissa bin** | The coefficient, normalized | Bin 6 (= 3.14) |
| **Unit** | Dollar / percent / shares | Dollar |

Now the GAT can learn that:
- `Revenue` (magnitude bin 11) behaves differently from `R&D` (bin 8)
- `Gross Margin` (percent) behaves differently from `Total Assets` (dollar)
- `3.14B` and `3.14M` share the same mantissa bin but different magnitude bins

### Human-Readable Summary

> **Old:** `log|số|` compresses everything into one number → GAT sees `$298B` and `$100M` as "both large numbers."
>
> **New:** Separate magnitude (bin 0–23), mantissa (0–19), and unit (dollar/percent/shares/ratio) → GAT sees them as *different levels of a financial hierarchy*, not just different values.

---

### Appendix: P0 Technical Details

#### Magnitude binning

```
Bin 0:  [0, 10⁻¹²)
Bin 1:  [10⁻¹², 10⁻¹¹)
...
Bin 12: [1, 10)
Bin 13: [10, 10²)
...
Bin 23: [10¹², ∞)
```

The financial range ($0 to hundreds of billions) maps to bins 4–11. A $1K value → bin 4. A $100B value → bin 11. The gap of 7 bins is the **order of magnitude difference** — the GAT now has 7 distinct embedding positions for this range.

#### Mantissa binning

```
value = coefficient × 10^floor(log10|value|)
```

For `$3.14B`: coefficient = 3.14. For `$314M`: coefficient = 3.14 (same coefficient, different magnitude). Both share the same mantissa bin → the GAT learns that "3.14" in different magnitudes has similar *structure*, while magnitude bin captures the scale.

#### Unit detection

```
Header/cell text contains "$" or "million" → dollar
Header/cell text ends with "%"            → percent
Header/cell text contains "shares"         → shares
Value < 1 and not a percentage              → ratio
Otherwise                                  → absolute
```

6 unit tokens. The GAT can learn that `percent` cells (margins) should not be compared directly with `dollar` cells (revenue).

#### Implementation

```python
class ScaleAwareNumericEncoder(nn.Module):
    def __init__(self, embed_dim):
        self.magnitude_embed = nn.Embedding(24, embed_dim // 3)   # bins 0–23
        self.mantissa_embed  = nn.Embedding(20, embed_dim // 3)   # bins 0–19
        self.unit_embed     = nn.Embedding(6,   embed_dim // 3)   # 6 units
        self.proj = nn.Linear(3 * embed_dim // 3, embed_dim)

    def forward(self, values, headers, cell_texts, device):
        # Compute 3 components per cell
        mag_bin   = [self._magnitude_bin(v)  for v in values]
        mant_bin  = [self._mantissa_bin(v)   for v in values]
        unit_id   = [self._unit_id(v, h, ct) for v, h, ct in zip(...)]
        # Embed → concat → project → [V, embed_dim]
```

**Added parameters:** `(24 + 20 + 6) × (embed_dim // 3) ≈ 40 × 341 ≈ 13.6K` trainable params — negligible vs 335M BGE backbone.

**Null handling:** Cells with `None`/`NaN` values → zero output vector (consistent with v1).

---

## P2 — Adaptive Relative Tolerance for Constraint Scoring

### The Problem

When a document's table violates accounting identities, `compute_constraint_score()` penalizes it. The penalty formula in v1:

```
score = exp(− residual / max(|tgt|, ε))
```

where `residual = |omega × src − tgt|`, and `tgt` is the target node value.

**The bug:** For mega-cap companies, `|tgt|` is enormous (in trillions), making the denominator `max(|tgt|, ε) ≈ |tgt|`. A **$10 billion accounting error** on a **$1 trillion total** produces:

```
V1 score = exp(− 10⁹ / 10¹²) = exp(− 0.01) ≈ 0.990
```

The table gets a **99% consistency score** despite a $10B discrepancy. The system thinks it's nearly perfect.

This matters because:
1. TAT-QA contains diverse tables — mega-cap balance sheets, mid-cap income statements, small-cap ratios
2. The same absolute error of `$10M` means completely different things at `$100B` scale vs `$100M` scale

### Why the Old Method Was Reasonable

`ε = 1e-4` (a tiny floor) was added to handle the case where `|tgt| = 0` (division by zero). The formula `exp(− residual / |tgt|)` works correctly for:

- **Small-value tables**: `|tgt| = 10⁶` → a $10K error → `exp(−0.01) ≈ 0.99` (reasonable tolerance for rounding)
- **Medium-value tables**: `|tgt| = 10⁹` → a $10M error → `exp(−0.01) ≈ 0.99`

The formula itself is sound. The issue is **domain mismatch**: accounting materiality thresholds are defined **relative to value**, not in absolute dollars.

> In financial auditing, "materiality" is set as a **percentage** of the relevant base (e.g., 5% of pre-tax income). A $1M error on a $20M company is material. A $1M error on a $300B company is rounding.

The original formula implicitly used **absolute tolerance** — appropriate for small tables, inappropriate for mega-cap tables.

### The Improvement

**Idea:** Replace absolute tolerance with **relative tolerance** — the tolerance scales with the target value.

```
V1: score = exp(− residual / max(|tgt|, ε))
V2: score = exp(− residual / (|tgt| × rel_tol + ε))
```

With `rel_tol = 1e-3` (0.1% relative tolerance):

| Scenario | \|tgt\| | residual | V1 score | V2 score |
|---|---|---|---|---|
| Perfect data | 1T | 0 | 1.000 | 1.000 |
| 0.1% rounding error | 1T | 1B | 0.990 | 0.368 ← tolerated |
| 10% gross violation | 1T | 100B | 0.905 | 0.000 ← detected |
| 0.1% rounding error | 1M | 1K | 0.990 | 0.368 ← same as above |
| 10% gross violation | 1M | 100K | 0.905 | 0.000 ← same as above |

V2 produces **consistent treatment** across magnitude scales. The same relative error gets the same score, regardless of company size.

### Tolerance Calibration

`rel_tol = 1e-3` (0.1%) is chosen based on financial materiality thresholds:

| rel_tol | Meaning | 0.1% error | 1% error | 10% error |
|---|---|---|---|---|
| `1e-4` (0.01%) | Very strict | `exp(-1) = 0.37` | `exp(-10) ≈ 0` | `exp(-100) ≈ 0` |
| **`1e-3` (0.1%)** | **Standard** | `exp(-0.1) = 0.90` | `exp(-1) = 0.37` | `exp(-10) ≈ 0` |
| `1e-2` (1%) | Lenient | `exp(-0.01) = 0.99` | `exp(-0.1) = 0.90` | `exp(-1) = 0.37` |
| `1e-1` (10%) | Very lenient | `exp(-0.001) ≈ 1.0` | `exp(-0.01) ≈ 1.0` | `exp(-0.1) = 0.90` |

`1e-3` means: typical financial rounding (0.1% — e.g., $300M round to $299.7M) is tolerated; anything above 1% is penalized significantly; 10% errors are near-zero score. This matches how auditors think about materiality.

### Human-Readable Summary

> **Old:** "A $10B error is $10B, whether it happens in a $1T company or a $100M company." → Fixed tolerance ignores scale.
>
> **New:** "A $10B error is 1% of a $1T company (tolerable rounding) but 10% of a $100M company (severe violation)." → Relative tolerance treats errors proportionally.

---

### Appendix: P2 Technical Details

#### Formula comparison

**V1 (original):**
```python
denom = max(abs(tgt.value), epsilon)  # floor by ε=1e-4
score = exp(-residual / denom)
# Works well for |tgt| < 1e9. Fails for |tgt| > 1e12.
```

**V2 (improved):**
```python
denom = abs(tgt.value) * relative_tolerance + epsilon  # relative + floor
score = exp(-residual / denom)
# Correctly penalizes large-value tables. epsilon only handles |tgt|≈0 edge case.
```

#### Zero-division guard

When `|tgt| ≈ 0` (ratio tables like `Net Income / Revenue = 0.15`):
- V1: `denom = max(0.15, 1e-4) ≈ 0.15` — works fine
- V2: `denom = 0.15 × 1e-3 + 1e-10 ≈ 1.5e-4` — same magnitude, works fine

The `epsilon` term (`+ 1e-10`) is a safety floor for the rare case of `|tgt| = 0`.

#### Why exp() specifically?

The exponential decay `exp(-x)` has properties that make it suitable:
1. `exp(0) = 1` (perfect constraint → score 1.0)
2. `exp(-1) ≈ 0.37` (1-unit error → moderate penalty)
3. `exp(-10) ≈ 4.5e-5` (10-unit error → near-zero score)
4. Bounded `[0, 1]` naturally, no clipping needed
5. Differentiable everywhere — gradients flow even for near-perfect constraints

Alternative: linear penalty `max(0, 1 − residual / denom)` — but this clips gradients at zero, causing training instability.

#### Edge case: ratio constraints

For `Revenue / Total Assets = ROA` (ratio constraint), `|tgt|` is already normalized (e.g., `0.05`). Then:
- V2: `denom = 0.05 × 1e-3 = 5e-5`
- A 10% relative error on ROA (`|error| = 0.005`) → `exp(-0.005 / 5e-5) = exp(-100) ≈ 0`
- Correctly penalizes ratio errors regardless of base value

#### Interaction with template matching

When `template_confidence < 0.5`, the system falls back to **positional edges** (`ω = 0`). In this case:
- `compute_constraint_score()` returns `score = 1.0` (no accounting constraints detected)
- V1 and V2 produce identical results
- The improvement only activates when **accounting constraints are matched** (roughly 60–80% of T²-RAGBench tables)

---

## Ablation Matrix

To measure the individual contribution of P0 and P2, run all four combinations:

```bash
# Baseline (original)
python -m gsr_cacl.train --dataset finqa --stage all --contr1 v1 --contr2 v1 --preset t4

# P0 only (scale-aware numeric, fixed epsilon)
python -m gsr_cacl.train --dataset finqa --stage all --contr1 v2 --contr2 v1 --preset t4

# P2 only (log-scale numeric, relative tolerance)
python -m gsr_cacl.train --dataset finqa --stage all --contr1 v1 --contr2 v2 --preset t4

# Both improvements
python -m gsr_cacl.train --dataset finqa --stage all --contr1 v2 --contr2 v2 --preset t4
```

Expected outcome on TAT-QA (diverse, multi-scale tables):

| Combination | Expected MRR@3 vs baseline |
|---|---|
| `--contr1 v1 --contr2 v1` | +0.00 (baseline) |
| `--contr1 v2 --contr2 v1` | +0.5–1.5% (P0: scale separation) |
| `--contr1 v1 --contr2 v2` | +0.5–2.0% (P2: relative tolerance) |
| `--contr1 v2 --contr2 v2` | +1.0–3.0% (combined) |

> **Note:** These are rough estimates. The actual numbers depend on the distribution of mega-cap vs mid-cap tables in each split. FinQA/ConvFinQA (mostly S&P 500) may show smaller gains from P2 than TAT-QA (diverse company sizes).
