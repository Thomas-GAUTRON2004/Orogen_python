"""Tectonic history model components ported from Orogen V2."""

from .arc_maturation import mature_arcs
from .backarc import open_back_arcs, rollback_cm_yr, steer_back_arc_strips
from .boundaries import (
    Motion,
    classify_contact_boundaries,
    classify_motion,
    feed_trench,
    near_land,
    vote_segments,
)
from .collision import (
    accrete_arcs,
    are_young_rift_partners,
    choose_sides,
    collect_contacts,
    dock_if_terrane,
    enclosed_sea_between_km2,
    forget_stale_collisions,
    is_rift_jitter,
    open_sea_between,
    recycle,
    recycle_small_arc,
    track_collisions,
    underthrust,
)
from .config import (
    CM_PER_YR_TO_KM_PER_MYR,
    CM_PER_YR_TO_RAD_PER_MYR,
    TECTONICS,
    MECHANISMS,
    mechanisms_off,
    mechanisms_on,
    only_mechanisms,
    make_tectonics_config,
    validate_tectonics_config,
)
from .contact import contact_at
from .crust import (
    BeltSide,
    CrustFlag,
    CrustRaster,
    CrustType,
    OrogenyType,
    VolcanicType,
)
from .dynamics import (
    balanced_rotation,
    clamp_speed,
    drag_tensor,
    soften_speed,
    solve3,
    update_plate_motion,
)
from .ending import ENDINGS, frames_ending, history_ending, in_stage
from .far_field import (
    evolve_far_field,
    far_take,
    feed_far_field,
    recent_push,
    record_push,
    spread_push,
)
from .forces import FORCE_KINDS, gather_boundary_loads
from .gather import clear_world_view, for_each_claim
from .geology import (
    GEO_FRAME_FIELDS, GEOLOGY_READS, GeoFlag,
    decode_age, decode_arc_myr, decode_cover_km, decode_exhum_km,
    decode_facies, decode_flood_age, decode_flood_myr, decode_unroof_km,
    encode_age, encode_arc_myr, encode_cover_km, encode_exhum_km,
    encode_facies, encode_flood_age, encode_flood_myr, encode_unroof_km,
    thicken_cover,
)
from .heights import belt_relief_km, broad_profile, build_heights
from .kinematics import (
    advance_plates,
    lat_lon_to_xyz,
    omega_from_euler_pole,
    pose_at_constant_omega,
    speed_cm_yr_at,
    surface_velocity,
    xyz_to_lat_lon,
)
from .mantle import dispersal_field, flow_of_cycle, mantle_flow_at, repel_at
from .occupancy import BinIndex, bin_index, mark_bin, may_have_crust_near
from .ops.merge_plates import merge_plates
from .orogeny import belt_shape, belt_width_km, obliquity_deg
from .plate import Plate, frame_rotation
from .plumes import birth_plume, update_plumes
from .polarity import overrides
from .reconstruct import reconstruct
from .resolve import candidate_cells, decide_birth, resolve
from .rifting import pace_of, rift_continents, search_junctions
from .segments import SEGMENT_KM, fracture_coordinate
from .simulation import choose_step, closing_rate, fastest_speed
from .spread_belt import spread_belt
from .stepper import step_world
from .subduction_init import founder_within_plates, initiate_subduction
from .supercontinent import is_continental, landmasses, track_cycle
from .thermal import (
    insulate_at_start,
    pick_plume_site,
    plume_heat_factor,
    update_mantle_heat,
)
from .world import BoundaryKind, EventKind, Provisional, TectonicWorld

__all__ = [
    "ENDINGS",
    "in_stage",
    "history_ending",
    "frames_ending",
    "BeltSide",
    "CrustFlag",
    "CrustRaster",
    "CrustType",
    "OrogenyType",
    "VolcanicType",
    "Plate",
    "frame_rotation",
    "BoundaryKind",
    "EventKind",
    "Provisional",
    "TectonicWorld",
    "BinIndex",
    "bin_index",
    "mark_bin",
    "may_have_crust_near",
    "clear_world_view",
    "for_each_claim",
    "overrides",
    "reconstruct",
    "contact_at",
    "resolve",
    "candidate_cells",
    "decide_birth",
    "Motion",
    "classify_motion",
    "vote_segments",
    "near_land",
    "feed_trench",
    "classify_contact_boundaries",
    "fastest_speed",
    "closing_rate",
    "choose_step",
    "belt_width_km",
    "belt_shape",
    "obliquity_deg",
    "spread_belt",
    "belt_relief_km",
    "broad_profile",
    "build_heights",
    "thicken_cover",
    "GeoFlag", "GEO_FRAME_FIELDS", "GEOLOGY_READS",
    "encode_cover_km", "decode_cover_km", "encode_exhum_km", "decode_exhum_km",
    "encode_unroof_km", "decode_unroof_km", "encode_flood_myr", "decode_flood_myr",
    "encode_arc_myr", "decode_arc_myr", "encode_age", "decode_age",
    "encode_flood_age", "decode_flood_age", "encode_facies", "decode_facies",
    "solve3",
    "drag_tensor",
    "balanced_rotation",
    "soften_speed",
    "clamp_speed",
    "update_plate_motion",
    "FORCE_KINDS",
    "gather_boundary_loads",
    "flow_of_cycle",
    "mantle_flow_at",
    "dispersal_field",
    "repel_at",
    "is_continental",
    "landmasses",
    "track_cycle",
    "step_world",
    "insulate_at_start",
    "update_mantle_heat",
    "plume_heat_factor",
    "pick_plume_site",
    "birth_plume",
    "update_plumes",
    "pace_of",
    "rift_continents",
    "search_junctions",
    "mature_arcs",
    "open_back_arcs",
    "rollback_cm_yr",
    "steer_back_arc_strips",
    "initiate_subduction",
    "founder_within_plates",
    "merge_plates",
    "feed_far_field",
    "evolve_far_field",
    "spread_push",
    "record_push",
    "far_take",
    "recent_push",
    "track_collisions",
    "accrete_arcs",
    "are_young_rift_partners",
    "choose_sides",
    "collect_contacts",
    "dock_if_terrane",
    "enclosed_sea_between_km2",
    "forget_stale_collisions",
    "is_rift_jitter",
    "open_sea_between",
    "recycle",
    "recycle_small_arc",
    "underthrust",
    "lat_lon_to_xyz",
    "xyz_to_lat_lon",
    "omega_from_euler_pole",
    "surface_velocity",
    "speed_cm_yr_at",
    "advance_plates",
    "pose_at_constant_omega",
    "CM_PER_YR_TO_KM_PER_MYR",
    "TECTONICS", "MECHANISMS", "mechanisms_off", "mechanisms_on", "only_mechanisms",
    "make_tectonics_config",
    "validate_tectonics_config",
    "CM_PER_YR_TO_RAD_PER_MYR",
    "SEGMENT_KM",
    "fracture_coordinate",
]
