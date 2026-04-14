"""Encoders for GSR-CACL: text, entity, GAT, and numeric."""

from gsr_cacl.encoders.positional import SinusoidalPositionalEncoding
from gsr_cacl.encoders.text_encoder import TextEncoder
from gsr_cacl.encoders.entity_encoder import EntityEncoder, SharedEncoder
from gsr_cacl.encoders.numeric_encoder import (
    ScaleAwareNumericEncoder,
    numeric_features_v1,
    build_numeric_encoder,
    NumericVersion,
)
from gsr_cacl.encoders.gat_layer import GATLayer
from gsr_cacl.encoders.gat_encoder import GATEncoder

__all__ = [
    # Text
    "TextEncoder",
    # Entity (NEW)
    "EntityEncoder",
    "SharedEncoder",
    # Numeric
    "ScaleAwareNumericEncoder",
    "numeric_features_v1",
    "build_numeric_encoder",
    "NumericVersion",
    # GAT
    "GATLayer",
    "GATEncoder",
    # Utils
    "SinusoidalPositionalEncoding",
]
