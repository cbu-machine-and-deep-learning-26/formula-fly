"""Degree-matched shuffled sparse projection (connectome-pattern control)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.interface import DEFAULT_FRAME_SHAPE, Eye, FrameArray, require_frame


def configuration_edges(
    out_degree: npt.NDArray[np.int64],
    in_degree: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build a directed stub-matching with exact in/out degrees (self-loops dropped)."""
    if int(out_degree.sum()) != int(in_degree.sum()):
        raise ValueError("in-degree sum must equal out-degree sum")
    sources = np.repeat(np.arange(out_degree.size, dtype=np.int64), out_degree)
    targets = np.repeat(np.arange(in_degree.size, dtype=np.int64), in_degree)
    rng.shuffle(targets)
    keep = sources != targets
    return sources[keep], targets[keep]


def degree_matched_shuffle(
    sources: npt.NDArray[np.int64],
    targets: npt.NDArray[np.int64],
    rng: np.random.Generator,
    n_swaps: int | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Swap destinations to scramble wiring while preserving in/out degrees."""
    src = np.array(sources, dtype=np.int64, copy=True)
    dst = np.array(targets, dtype=np.int64, copy=True)
    n_edges = src.size
    if n_edges < 2:
        return src, dst
    swaps = 10 * n_edges if n_swaps is None else int(n_swaps)
    for _ in range(swaps):
        i, j = (int(x) for x in rng.integers(0, n_edges, size=2))
        if i == j:
            continue
        if src[i] == src[j] or dst[i] == dst[j]:
            continue
        if src[i] == dst[j] or src[j] == dst[i]:
            continue
        dst[i], dst[j] = dst[j], dst[i]
    return src, dst


def degrees(
    sources: npt.NDArray[np.int64],
    targets: npt.NDArray[np.int64],
    n_src: int,
    n_dst: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    out_d = np.bincount(sources, minlength=n_src).astype(np.int64)
    in_d = np.bincount(targets, minlength=n_dst).astype(np.int64)
    return out_d, in_d


class ShuffledConnectomeEye(Eye):
    """Sparse random bipartite map with a degree-matched rewired copy of itself.

    Phase 0 uses a synthetic degree sequence, not FlyWire. The FlyWire swap
    lands with the real optic-lobe graph (issue on matched-size control eyes).
    """

    def __init__(
        self,
        output_size: int = 256,
        input_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        seed: int = 0,
        mean_degree: int = 8,
    ) -> None:
        if output_size < 1:
            raise ValueError("output_size must be positive")
        self._input_shape = input_shape
        self._output_size = int(output_size)
        n_in = int(np.prod(input_shape))
        rng = np.random.default_rng(seed)
        # Exact out-degree = mean_degree for every input; random targets (configuration-like).
        sources = np.repeat(np.arange(n_in, dtype=np.int64), int(mean_degree))
        targets = rng.integers(0, self._output_size, size=sources.size, dtype=np.int64)
        self._n_in = n_in
        self.sources_original = sources
        self.targets_original = targets.copy()
        # A short rewire is enough to scramble pattern; tests use small graphs + many swaps.
        self.sources, self.targets = degree_matched_shuffle(
            sources, targets, rng, n_swaps=min(2000, int(sources.size))
        )
        n_edges = self.sources.size
        self._weights = rng.normal(0.0, 1.0 / np.sqrt(max(mean_degree, 1)), size=n_edges).astype(
            np.float32
        )

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_size(self) -> int:
        return self._output_size

    def encode(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        pixels = require_frame(frame, self._input_shape) / 255.0
        flat = pixels.reshape(-1)
        out = np.zeros(self._output_size, dtype=np.float32)
        np.add.at(out, self.targets, flat[self.sources] * self._weights)
        return out
