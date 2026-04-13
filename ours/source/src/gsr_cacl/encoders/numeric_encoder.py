"""Numeric encoding for financial table cells.

Two versions:
    v1 (default): log-scale + sign + is_zero + magnitude bucket  [4 features]
    v2:           multi-resolution: magnitude bin + mantissa + unit embeddings

P0 improvement: v2 addresses the issue where log-scale collapses large-value
differences. For example, |v|=100B (log≈26) vs |v|=100M (log≈18) differ by only
~8 log units despite being 1000× apart in actual scale — insufficient for the GAT
to distinguish a Fortune 500 balance sheet from a mid-cap income statement.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Unit detection
# ---------------------------------------------------------------------------

_UNIT_KEYWORDS: dict[str, str] = {
    # Currency
    "$": "dollar", "usd": "dollar", "eur": "dollar", "gbp": "dollar",
    # Percentage
    "%": "percent", "percentage": "percent", "rate": "percent",
    # Scale suffixes embedded in header text
    "million": "M", "billion": "B", "thousand": "K",
    "mm": "M", "bn": "B", "shr": "shares",
}


def _detect_unit(header: str, cell_text: str = "") -> str:
    """
    Detect the unit of a cell based on its header and cell text.
    Returns one of: "dollar", "percent", "shares", "ratio", "absolute"
    """
    combined = (header + " " + cell_text).lower()

    for kw, unit in _UNIT_KEYWORDS.items():
        if kw in combined:
            return unit

    # Check for percentage in cell text
    if isinstance(cell_text, str) and cell_text.strip().endswith("%"):
        return "percent"

    # Check for scale suffix
    if isinstance(cell_text, str):
        s = cell_text.strip()
        if s.endswith(("B", "b")) and _is_number(s[:-1]):
            return "dollar"
        if s.endswith(("M", "m")) and _is_number(s[:-1]):
            return "dollar"
        if s.endswith(("K", "k")) and _is_number(s[:-1]):
            return "dollar"

    return "absolute"


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("$", ""))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# V1: Original log-scale encoding [4 features]
# ---------------------------------------------------------------------------

def numeric_features_v1(value: Optional[float], device: torch.device) -> torch.Tensor:
    """
    Encode a numeric value into a [4] feature vector.

    Features: [log|v|, sign, is_zero, magnitude_bucket]
    - log|v|: log1p of absolute value — captures order of magnitude
    - sign: 1.0 for positive, -1.0 for negative
    - is_zero: 1.0 if |v| < 1e-8
    - magnitude_bucket: 0–4 (tiny/small/medium/large/huge)

    Limitation: For |v|=100B→log≈26 vs |v|=100M→log≈18, the difference is only
    ~8 log units despite being 1000× apart in real scale. The GAT cannot
    meaningfully distinguish these values from the log feature alone.
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return torch.zeros(4, device=device)
    sign = 1.0 if value >= 0 else -1.0
    is_zero = 1.0 if abs(value) < 1e-8 else 0.0
    log_abs = math.log1p(abs(value))
    # magnitude bucket: 0=tiny(<1), 1=small(<1K), 2=medium(<1M), 3=large(<1B), 4=huge(>=1B)
    abs_v = abs(value)
    if abs_v < 1:
        bucket = 0.0
    elif abs_v < 1e3:
        bucket = 1.0
    elif abs_v < 1e6:
        bucket = 2.0
    elif abs_v < 1e9:
        bucket = 3.0
    else:
        bucket = 4.0
    return torch.tensor([log_abs, sign, is_zero, bucket / 4.0], device=device)


# ---------------------------------------------------------------------------
# V2: Multi-resolution scale-aware encoding
# ---------------------------------------------------------------------------

class ScaleAwareNumericEncoder(nn.Module):
    """
    Multi-resolution numeric encoding for financial tables.

    Decomposes a number into three disentangled components:
    1. Magnitude bin (log-scale): which power of 10 does |v| fall in?
       This is the ORDER OF MAGNITUDE — e.g. 100B → bin 11 (10^11), 100M → bin 8 (10^8).
       Gap of 3 bins = 1000× difference, clearly distinguished.
    2. Mantissa: value / 10^floor(log10|v|) → normalized to [0.1, 10)
       This captures the coefficient — e.g. 3.14B vs 1.00B vs 9.99B.
    3. Sign + unit awareness: positive/negative, and whether it's $/%/shares.

    Embedding dimensions:
        magnitude_embed:  [NUM_MAG_BINS, embed_dim // 3]
        mantissa_embed:   [NUM_MANTISSA_BINS, embed_dim // 3]
        unit_embed:       [NUM_UNITS, embed_dim // 3]

    Financial rationale:
        - A revenue table has values from $1K (R&D) to $300B (total revenue).
          V1 log-scale treats both as just "large numbers" with log 7–26.
          V2 separates magnitude (which bin: 3=thousands vs 11=hundreds of billions)
          from mantissa (3.14 vs 0.05 within each bin).
        - Ratio tables (margins, ROE) use percentages: 0.05 vs 0.25 are meaningfully
          different and should be distinguished even though both are <1.
        - Unit awareness lets the model learn that "$" cells behave differently from "%" cells.
    """

    NUM_MAG_BINS = 24       # 10^-12 to 10^+12 (covers all financial numbers)
    NUM_MANTISSA_BINS = 20  # [0.05, 0.10), [0.10, 0.15), ..., [0.95, 1.0), [1.0, 1.05), ...
    NUM_UNITS = 6           # dollar, percent, shares, ratio, absolute, negative

    UNIT_TO_ID: dict[str, int] = {
        "dollar": 0,
        "percent": 1,
        "shares": 2,
        "ratio": 3,
        "absolute": 4,
        "negative": 5,
    }

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        d = embed_dim // 3

        self.magnitude_embed = nn.Embedding(self.NUM_MAG_BINS, d)
        self.mantissa_embed = nn.Embedding(self.NUM_MANTISSA_BINS, d)
        self.unit_embed = nn.Embedding(self.NUM_UNITS, d)

        # Projection to merge three components → embed_dim
        self.proj = nn.Sequential(
            nn.Linear(3 * d, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def _compute_magnitude_bin(self, value: float) -> int:
        """
        Map |value| to a magnitude bin [0, NUM_MAG_BINS-1].

        Bin 0:  [0, 10^-12)
        Bin 1:  [10^-12, 10^-11)
        ...
        Bin 12: [1, 10)
        Bin 13: [10, 10^2)
        ...
        Bin 24: [10^12, ∞)
        """
        if value < 1e-12:
            return 0
        if value >= 1e12:
            return self.NUM_MAG_BINS - 1
        # log10(|v|) in range [-12, 12]
        log_val = math.log10(value)
        bin_idx = int(log_val + 12)  # shift to [0, 24]
        return min(max(bin_idx, 0), self.NUM_MAG_BINS - 1)

    def _compute_mantissa_bin(self, value: float) -> int:
        """
        Map value/10^floor(log10|v|) to a mantissa bin [0, NUM_MANTISSA_BINS-1].

        Normalizes the coefficient: 3.14B → mantissa 3.14 → bin covering [3.10, 3.15)
        This is the relative position within the order of magnitude.

        For negatives: use |value| and mark unit as "negative".
        """
        abs_v = abs(value)
        if abs_v < 1e-12:
            return 0  # zero → first bin

        # mantissa = |value| / 10^floor(log10|v|) ∈ [1.0, 10.0)
        log_mantissa = math.log10(abs_v)
        floor_log = math.floor(log_mantissa)
        mantissa = abs_v / (10 ** floor_log)  # ∈ [1.0, 10.0)

        # Map [1.0, 10.0) → [0, NUM_MANTISSA_BINS)
        # bin_width = 9.0 / NUM_MANTISSA_BINS ≈ 0.45 per bin
        bin_idx = int((mantissa - 1.0) / 9.0 * self.NUM_MANTISSA_BINS)
        return min(max(bin_idx, 0), self.NUM_MANTISSA_BINS - 1)

    def _compute_unit_id(self, value: float, header: str, cell_text: str) -> int:
        unit = _detect_unit(header, cell_text)
        if value < 0:
            return self.UNIT_TO_ID["negative"]
        return self.UNIT_TO_ID.get(unit, self.UNIT_TO_ID["absolute"])

    def forward(
        self,
        values: list[float | None],
        headers: list[str],
        cell_texts: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode a batch of numeric values.

        Args:
            values:       list of float values (or None for non-numeric cells)
            headers:      list of column header strings
            cell_texts:   list of raw cell text strings
            device:       torch device
        Returns:
            embeddings:   [len(values), embed_dim]
        """
        V = len(values)
        mag_bins = []
        mant_bins = []
        unit_ids = []
        mask = []  # 1.0 = valid, 0.0 = null

        for i in range(V):
            v = values[i]
            h = headers[i] if i < len(headers) else ""
            ct = cell_texts[i] if i < len(cell_texts) else ""

            if v is None or math.isnan(v) or math.isinf(v):
                mag_bins.append(0)
                mant_bins.append(0)
                unit_ids.append(self.UNIT_TO_ID["absolute"])
                mask.append(0.0)
            else:
                mag_bins.append(self._compute_magnitude_bin(abs(v)))
                mant_bins.append(self._compute_mantissa_bin(v))
                unit_ids.append(self._compute_unit_id(v, h, ct))
                mask.append(1.0)

        mag_t = torch.tensor(mag_bins, dtype=torch.long, device=device)
        mant_t = torch.tensor(mant_bins, dtype=torch.long, device=device)
        unit_t = torch.tensor(unit_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

        mag_emb = self.magnitude_embed(mag_t)      # [V, d]
        mant_emb = self.mantissa_embed(mant_t)      # [V, d]
        unit_emb = self.unit_embed(unit_t)          # [V, d]

        emb = torch.cat([mag_emb, mant_emb, unit_emb], dim=-1)  # [V, 3d]
        emb = self.proj(emb)                        # [V, embed_dim]

        # Zero out null cells (consistent with v1 behavior)
        emb = emb * mask_t.unsqueeze(-1)

        return emb

    def forward_single(
        self,
        value: float | None,
        header: str = "",
        cell_text: str = "",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode a single numeric value."""
        if device is None:
            device = next(self.parameters()).device
        return self.forward([value], [header], [cell_text], device).squeeze(0)


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

NumericVersion = Literal["v1", "v2"]


def build_numeric_encoder(
    version: NumericVersion,
    embed_dim: int,
) -> tuple[nn.Module | None, callable]:
    """
    Build a numeric encoder and return (encoder_module, feature_fn).

    For v1: encoder_module=None, feature_fn=numeric_features_v1 (stateless)
    For v2: encoder_module=ScaleAwareNumericEncoder, feature_fn=None

    Usage in GATEncoder:
        if module is not None:
            cell_emb = module.forward(values, headers, cell_texts, device)
        else:
            cell_emb = torch.stack([fn(v, device) for v in values])

    Returns:
        (encoder_module, feature_fn) — one is None, one is valid
    """
    if version == "v1":
        return None, numeric_features_v1
    elif version == "v2":
        return ScaleAwareNumericEncoder(embed_dim), None
    else:
        raise ValueError(f"Unknown numeric version: {version}. Choose 'v1' or 'v2'.")
