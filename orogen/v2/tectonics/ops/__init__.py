"""Tectonic plate operations ported from Orogen V2."""

from .birth_ocean_plate import birth_ocean_plate, ocean_plate_name
from .merge_plates import merge_plates
from .plate_accretion import accrete_terrane, close_seas_between, landmass_of
from .plate_transfer import free_cell_beside, transfer_cells
from .rift import (
    choose_junction,
    names_after_split,
    plume_junctions,
    pull_axis_of,
    split_plate,
    try_rift,
)
from .rift_partition import (
    elongation_of,
    exterior_sea,
    masked_cells,
    partition,
    sides_of,
)
from .rift_path import find_rift, grain_of, rift_context
from .scrape_arc import scrape_arc

__all__ = [
    "elongation_of",
    "exterior_sea",
    "masked_cells",
    "partition",
    "sides_of",
    "find_rift",
    "grain_of",
    "rift_context",
    "choose_junction",
    "split_plate",
    "pull_axis_of",
    "names_after_split",
    "try_rift",
    "plume_junctions",
    "scrape_arc",
    "free_cell_beside",
    "transfer_cells",
    "birth_ocean_plate",
    "ocean_plate_name",
    "accrete_terrane",
    "close_seas_between",
    "landmass_of",
    "merge_plates",
]
