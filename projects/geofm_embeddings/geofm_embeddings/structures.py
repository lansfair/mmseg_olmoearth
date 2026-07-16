from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from torch import Tensor


EmbeddingMode = Literal["global", "dense"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Static input and output capabilities declared by a GeoFM adapter."""

    supported_modalities: frozenset[str]
    required_modalities: frozenset[str] = field(default_factory=frozenset)
    supports_global: bool = True
    supports_dense: bool = True
    supports_multitemporal: bool = True
    supports_multimodal: bool = True
    native_stride: int | None = None

    def validate(self, modalities: set[str], mode: EmbeddingMode) -> None:
        missing = self.required_modalities - modalities
        if missing:
            raise ValueError(
                "Missing required modalities: " + ", ".join(sorted(missing))
            )

        unsupported = modalities - self.supported_modalities
        if unsupported:
            raise ValueError(
                "Unsupported modalities: " + ", ".join(sorted(unsupported))
            )

        if len(modalities) > 1 and not self.supports_multimodal:
            raise ValueError(
                "This adapter accepts only one modality per forward pass."
            )
        if mode == "global" and not self.supports_global:
            raise ValueError("This adapter does not provide global embeddings.")
        if mode == "dense" and not self.supports_dense:
            raise ValueError("This adapter does not provide dense embeddings.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported_modalities": sorted(self.supported_modalities),
            "required_modalities": sorted(self.required_modalities),
            "supports_global": self.supports_global,
            "supports_dense": self.supports_dense,
            "supports_multitemporal": self.supports_multitemporal,
            "supports_multimodal": self.supports_multimodal,
            "native_stride": self.native_stride,
        }


@dataclass
class EmbeddingResult:
    """A tensor plus the provenance needed to export it safely."""

    tensor: Tensor
    mode: EmbeddingMode
    model_family: str
    model_variant: str
    modalities: tuple[str, ...]
    native_stride: int | None = None
    pooling: str | None = None
    normalized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_ndim = 2 if self.mode == "global" else 4
        if self.tensor.ndim != expected_ndim:
            raise ValueError(
                f"{self.mode} embeddings must have {expected_ndim} dimensions, "
                f"got shape {tuple(self.tensor.shape)}"
            )

    @property
    def embedding_dim(self) -> int:
        return int(self.tensor.shape[1])

    def manifest_metadata(self) -> dict[str, Any]:
        out = {
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "modalities": list(self.modalities),
            "mode": self.mode,
            "embedding_dim": self.embedding_dim,
            "native_stride": self.native_stride,
            "pooling": self.pooling,
            "l2_normalized": self.normalized,
            "dtype": str(self.tensor.dtype).removeprefix("torch."),
        }
        out.update(self.metadata)
        return out
