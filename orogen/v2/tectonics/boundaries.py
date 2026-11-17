"""Boundary-motion primitives ported from Orogen V2 step/boundaries.js.

This module currently ports the deterministic classification and trench bookkeeping
helpers. The full world pass also depends on orogeny and far-field deformation and
is intentionally kept separate until those systems are ported.
"""
from __future__ import annotations

from .crust import CrustFlag, CrustType
from ..core.mesh_bfs import reaches_within


class Motion:
    LOCKED = 0
    OPENING = 1
    CLOSING = 2
    SLIDING = 3


def classify_motion(closing: float, shear: float, active_speed: float, oblique_limit: float) -> int:
    """Classify relative contact motion from normal closing rate and shear."""
    normal_dominates = abs(closing) > oblique_limit * shear
    if closing < -active_speed and normal_dominates:
        return Motion.OPENING
    if closing > active_speed and normal_dominates:
        return Motion.CLOSING
    return Motion.SLIDING if shear > active_speed else Motion.LOCKED


def vote_segments(world, contacts: list[dict]) -> None:
    """Vote one subduction polarity per connected ocean-ocean contact segment."""
    mesh, overrider = world.mesh, world.oceanOverrider
    overrider.fill(-1)
    if not contacts:
        return
    by_pair: dict[str, list[dict]] = {}
    for contact in contacts:
        by_pair.setdefault(contact["key"], []).append(contact)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    for group in by_pair.values():
        at = {contact["w"]: contact for contact in group}
        seen: set[int] = set()
        for start in group:
            if start["w"] in seen:
                continue
            run = [start]
            seen.add(start["w"])
            index = 0
            while index < len(run):
                current = run[index]
                index += 1
                for k in range(int(offsets[current["w"]]), int(offsets[current["w"] + 1])):
                    neighbour = at.get(int(adjacency[k]))
                    if neighbour is not None and neighbour["w"] not in seen:
                        seen.add(neighbour["w"])
                        run.append(neighbour)
            score = sum(c["weight"] if c["prefersLower"] else -c["weight"] for c in run)
            if score == 0:
                winner = run[0]["lowerId"] if run[0]["prefersLower"] else run[0]["upperId"]
            else:
                winner = run[0]["lowerId"] if score > 0 else run[0]["upperId"]
            for contact in run:
                overrider[contact["w"]] = winner


def near_land(plate, mesh, cell: int, config: dict) -> bool:
    """Whether non-ocean crust is within the subduction apron reach."""
    hops = round((config["apronKm"] + 2 * mesh.edge_km) / mesh.edge_km)
    crust = plate.crust
    return reaches_within(
        mesh, cell, hops,
        lambda c: bool(crust.mask[c]),
        lambda c: int(crust.type[c]) != CrustType["OCEAN"],
    )


def feed_trench(world, world_cell: int, crust, cell: int, subducted_km: float) -> bool:
    """Establish/extend a trench only when this cell or a neighbour consumed crust."""
    mesh = world.mesh
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    if not (int(crust.flags[cell]) & CrustFlag["TRENCH"]):
        fed = bool(world.consumedAt[world_cell])
        for k in range(int(offsets[world_cell]), int(offsets[world_cell + 1])):
            if fed:
                break
            fed = bool(world.consumedAt[int(adjacency[k])])
        if not fed:
            return False
        crust.flags[cell] |= CrustFlag["TRENCH"]
    crust.record_subduction(cell, subducted_km)
    return subducted_km > 0
