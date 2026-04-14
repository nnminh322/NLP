"""EntitySupConLoss: Supervised Contrastive Learning for Entity Understanding.

Implements §4.4 of the GSR-CACL proposal.

Key insight:
    Entity equivalence is a DISCRETE property (same/different entity) but must
    be learned from CONTINUOUS representations. This is why we need contrastive
    learning instead of simple classification.

    Positive pairs: ("Apple", "Apple Inc."), ("AAPL", "Apple Inc."), ("MSFT", "Microsoft")
    Negative pairs: all pairs where entity labels differ

Why SupConLoss (Khosla et al., NeurIPS 2020) and not Triplet Loss:
    - Triplet loss: only 1 anchor-positive-negative tuple per update
    - SupCon loss: ALL positive pairs in the same batch contribute to the gradient
    - SupCon produces more compact clusters (compact cluster = better entity resolution)

Equation (Eq.13 in proposal):
    L = -log ( Σ_{j∈P(i)} exp(cos(z_i, z_j)/τ) )
                        --------------------------------
                        Σ_{k=1}^{B} exp(cos(z_i, z_k)/τ)

where P(i) = {j | label(i) = label(j), j ≠ i} are all positives in the batch.

Why temperature τ = 0.07:
    Small τ makes the distribution sharper — embeddings must be VERY close to
    ALL positives and VERY far from ALL negatives. Large τ is more forgiving.
    τ = 0.07 is the standard in CLIP/SimCLR and works well for entity resolution.

Gradient flow:
    ∂L/∂θ flows through entity_encoder → shared BGE backbone → improves BOTH
    entity embeddings AND text embeddings (joint benefit for retrieval).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntitySupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for entity understanding.

    Given a batch of entity mentions with canonical labels, this loss:
        (1) Pulls embeddings of same-entity mentions TOGETHER in embedding space
        (2) Pushes embeddings of different-entity mentions APART
        (3) Uses ALL positive pairs in the batch (not just 1 pair like triplet)

    Args:
        temperature: Sharpness parameter (τ in paper). Default 0.07.
                     Smaller = stricter clustering, larger = softer clustering.

    Usage:
        loss_fn = EntitySupConLoss(temperature=0.07)
        entity_embeddings = entity_encoder(companies, years, sectors, device)
        labels = batch_entity_labels  # canonical entity labels
        loss = loss_fn(entity_embeddings, labels)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SupCon loss for entity embeddings.

        Args:
            embeddings: [B, d_e] entity embeddings (must be L2-normalized)
            labels: [B] canonical entity labels (can be string or int)

        Returns:
            Scalar loss (mean over batch)
        """
        # Normalize embeddings (defensive — should already be normalized)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute pairwise cosine similarity: [B, B]
        # For normalized vectors, cosine similarity = dot product
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Compute loss for each anchor i
        # We use the standard SupCon formulation (Khosla et al., 2020)

        # Convert labels to tensor if needed (for comparison)
        if not torch.is_tensor(labels):
            # labels are strings or ints — convert to numeric
            label_keys = {}
            unique_labels = []
            for label in labels:
                key = str(label)
                if key not in label_keys:
                    label_keys[key] = len(label_keys)
                    unique_labels.append(key)

            numeric_labels = torch.tensor(
                [label_keys[str(l)] for l in labels],
                device=embeddings.device,
                dtype=torch.long,
            )
        else:
            numeric_labels = labels

        # Create positive mask: same label (excluding self)
        # positive_mask[i,j] = 1 if labels[i] == labels[j] and i != j
        B = embeddings.size(0)
        labels_equal = numeric_labels.unsqueeze(1) == numeric_labels.unsqueeze(0)  # [B, B]
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_equal & eye_mask  # [B, B]

        # For numerical stability: subtract max per row
        max_sim = similarity.max(dim=1, keepdim=True)[0]
        similarity_stable = similarity - max_sim

        # exp of similarity
        exp_sim = torch.exp(similarity_stable)

        # FIX ISSUE 7: Denominator should EXCLUDE self (i == j)
        # Per Khosla et al. (NeurIPS 2020), the denominator sum is over all k ≠ i
        # Self-similarity (i=j) is always 1/τ → very large, would dominate denominator
        denominator = (exp_sim * eye_mask.float()).sum(dim=1, keepdim=True)  # [B, 1]

        # Numerator: sum of exp over POSITIVE j only
        # Use masked sum: only positive pairs contribute
        positive_exp_sum = (exp_sim * positive_mask.float()).sum(dim=1, keepdim=True)  # [B, 1]

        # SupCon loss per anchor: -log(numerator / denominator)
        per_sample_loss = -torch.log(positive_exp_sum / (denominator + 1e-10) + 1e-10)

        # Average over positive pairs per anchor (not over all batch)
        # This prevents large batches from having small loss artificially
        positive_count = positive_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        per_sample_loss = per_sample_loss / positive_count

        return per_sample_loss.mean()


# ---------------------------------------------------------------------------
# Entity Registry: labeled positive pairs for supervised training
# ---------------------------------------------------------------------------

class EntityRegistry:
    """
    Registry of known entity equivalences for supervised contrastive learning.

    Sources:
        (1) T²-RAGBench metadata: same (company, year, sector) → positive pair
        (2) SEC CIK Registry: ticker → company name mapping (CIK: 0000320193)
        (3) SEC EDGAR tickers JSON: public ticker → company name
        (4) In-batch hard negatives: other companies in the same batch

    This registry provides the labeled pairs needed for EntitySupConLoss.
    During training, pairs from this registry form the positive set P(i).
    """

    # Pre-built SEC CIK mapping for major US-listed companies
    # Coverage: S&P 500 companies (the vast majority of T²-RAGBench corpus)
    CIK_MAPPING: dict[str, str] = {
        # Technology
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "GOOG": "Alphabet Inc.",
        "AMZN": "Amazon.com, Inc.",
        "META": "Meta Platforms, Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla, Inc.",
        "AVGO": "Broadcom Inc.",
        # Finance
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corporation",
        "WFC": "Wells Fargo & Company",
        "GS": "The Goldman Sachs Group, Inc.",
        "MS": "Morgan Stanley",
        "V": "Visa Inc.",
        "MA": "Mastercard Incorporated",
        # Industrial
        "CAT": "Caterpillar Inc.",
        "HON": "Honeywell International Inc.",
        "UNP": "Union Pacific Corporation",
        "RTX": "RTX Corporation",
        "LMT": "Lockheed Martin Corporation",
        "BA": "The Boeing Company",
        # Consumer
        "PG": "The Procter & Gamble Company",
        "KO": "The Coca-Cola Company",
        "PEP": "PepsiCo, Inc.",
        "COST": "Costco Wholesale Corporation",
        "WMT": "Walmart Inc.",
        "HD": "The Home Depot, Inc.",
        "MCD": "McDonald's Corporation",
        "NKE": "NIKE, Inc.",
        # Energy
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "COP": "ConocoPhillips",
        # Healthcare
        "LLY": "Eli Lilly and Company",
        "UNH": "UnitedHealth Group Incorporated",
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "ABBV": "AbbVie Inc.",
        "MRK": "Merck & Co., Inc.",
        "TMO": "Thermo Fisher Scientific Inc.",
        "ABT": "Abbott Laboratories",
        "AMGN": "Amgen Inc.",
        "GILD": "Gilead Sciences, Inc.",
        "ISRG": "Intuitive Surgical, Inc.",
        "MDT": "Medtronic plc",
        "SYK": "Stryker Corporation",
        "BMY": "Bristol-Myers Squibb Company",
        "CVS": "CVS Health Corporation",
        # Communication
        "CMCSA": "Comcast Corporation",
        "VZ": "Verizon Communications Inc.",
        "T": "AT&T Inc.",
        "TMUS": "T-Mobile US, Inc.",
        "DIS": "The Walt Disney Company",
        # Materials
        "APD": "Air Products and Chemicals, Inc.",
        "LIN": "Linde plc",
        "SHW": "The Sherwin-Williams Company",
        "NEM": "Newmont Corporation",
        # Real Estate
        "AMT": "American Tower Corporation",
        "PLD": "Prologis, Inc.",
        "EQIX": "Equinix, Inc.",
        # Utilities
        "NEE": "NextEra Energy, Inc.",
        "DUK": "Duke Energy Corporation",
        "SO": "The Southern Company",
        # ETF / Index (some Q&A involve these)
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "IWM": "iShares Russell 2000 ETF",
    }

    # Ticker → company canonical name (for creating positive pairs)
    TICKER_ALIASES: dict[str, set[str]] = {
        "AAPL": {"Apple Inc.", "Apple", "Apple Computer, Inc.", "APPLE INC.", "Apple Inc"},
        "MSFT": {"Microsoft Corporation", "Microsoft", "MSFT", "MICROSOFT CORP"},
        "GOOGL": {"Alphabet Inc.", "Alphabet", "GOOGL", "GOOGLE", "Alphabet Inc"},
        "AMZN": {"Amazon.com, Inc.", "Amazon", "AMZN", "AMAZON COM INC"},
        "META": {"Meta Platforms, Inc.", "Meta", "Facebook", "Facebook, Inc.", "Meta Platforms"},
        "NVDA": {"NVIDIA Corporation", "NVIDIA", "NVDA", "NVIDIA Corp"},
        "TSLA": {"Tesla, Inc.", "Tesla", "TSLA", "TESLA INC"},
    }

    def __init__(self):
        self._company_cache: dict[str, str] = {}

    def get_canonical_name(self, ticker_or_name: str) -> str | None:
        """Look up the canonical company name for a ticker or partial name."""
        key = ticker_or_name.upper().strip()
        if key in self.CIK_MAPPING:
            return self.CIK_MAPPING[key]
        if key in self._company_cache:
            return self._company_cache[key]
        return None

    def are_same_entity(self, name1: str, name2: str) -> bool:
        """
        Check if two company references point to the same entity.

        Handles:
            - Exact match (case-insensitive)
            - Ticker → canonical name matching
            - Partial substring matching ("Apple" in "Apple Inc.")
        """
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # Exact match
        if n1 == n2:
            return True

        # FIX ISSUE 11: Ticker resolution — both must resolve AND be the same entity
        ticker1 = self.get_canonical_name(name1)
        ticker2 = self.get_canonical_name(name2)
        # Only return True if BOTH resolve AND they resolve to the SAME canonical name
        if ticker1 and ticker2 and ticker1 == ticker2:
            return True

        # Substring match (fallback heuristic — less reliable)
        if n1 in n2 or n2 in n1:
            return True

        return False

    def get_positive_pairs(self, company_names: list[str]) -> list[tuple[int, int]]:
        """
        Given a list of company names, return indices of positive pairs.

        Two mentions are positive if they refer to the same entity.
        Used to construct the positive mask P(i) for SupCon loss.
        """
        pairs = []
        for i, name1 in enumerate(company_names):
            for j, name2 in enumerate(company_names):
                if i >= j:
                    continue
                if self.are_same_entity(name1, name2):
                    pairs.append((i, j))
        return pairs

    def build_entity_labels(self, company_names: list[str]) -> list[str]:
        """
        Assign canonical entity labels to a list of company names.

        Returns:
            List of canonical entity names (for SupCon label assignment).
        """
        labels = []
        for name in company_names:
            canonical = self.get_canonical_name(name)
            if canonical is None:
                # Fall back to normalized form
                canonical = name.strip()
            labels.append(canonical)
        return labels
