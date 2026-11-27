"""Precise configuration schema shared by initial-world geometry builders."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

ConfigValue: TypeAlias = str | int | float | bool | None | list["ConfigValue"] | dict[str, "ConfigValue"]

NumericRange = list[int | float] | tuple[int | float, int | float]


class InitialConfig(TypedDict, total=False):
    """Subset of tectonic configuration consumed by craton and landmass builders."""

    cratonRiftSteps: int
    cratonRiftStepKm: NumericRange
    cratonRiftDepth: NumericRange
    cratonRiftEdges: NumericRange
    cratonMaxDiameterKm: int | float
    cratonMaxAspect: int | float
    cratonLobes: int | float
    cratonMinCompact: int | float
    cratonCount: int
    cratonShare: int | float
    continentalFraction: int | float
    cratonSizeRatio: int | float
    cratonSizeSkew: int | float
    cratonMinGapKm: int | float
    cratonEdgeNoiseKm: int | float
    cratonSiblingChance: int | float
    clusterBlocks: NumericRange
    startMaxLatitudeDeg: int | float
    clusterElongation: int | float
    clusterApart: NumericRange
    clusterGapVariety: int | float
    clusterLooseness: int | float
    startArcKm: NumericRange
    startArcRadiusKm: NumericRange
    startStepKm: NumericRange
    startStepOffsetKm: NumericRange
    cratonGrainNoiseKm: NumericRange
    coastUnionSoftnessKm: int | float
    terraneBorderWanderKm: int | float
    sutureHalfWidthKm: int | float
    startCoreKm: int | float
    startGrowthBias: int | float
    startSeamWanderKm: int | float
    startRingKm: NumericRange
    cratonRiftNoise: int | float
    startCoastDetailKm: int | float
    coastReliefKm: int | float
    cratonMinCoastKm: int | float
    bridgeHalfWidthKm: int | float
