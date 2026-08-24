"""Small, solver-independent data model for the PMMA blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockSpec:
    name: str
    origin: tuple[float, float]
    dimensions: tuple[float, float]
    tag_prefix: int


@dataclass(frozen=True)
class Material:
    name: str
    rho: float
    E: float
    nu: float

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lmbda(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def cp(self) -> float:
        return math.sqrt((self.lmbda + 2.0 * self.mu) / self.rho)


@dataclass(frozen=True)
class FrictionReference:
    mu_s: float
    mu_k: float
    d_c: float


@dataclass(frozen=True)
class LoadingReference:
    simulation_time: float
    time_factor: float
    normal_stress: float
    rise_fraction: float
    tau_k_start_fraction: float
    normal_dir: int
    slave_surface: str
    master_surface: str


@dataclass(frozen=True)
class PMMAModelInput:
    moving: BlockSpec
    stationary: BlockSpec
    materials: dict[str, Material]
    friction: FrictionReference
    simulation: LoadingReference
