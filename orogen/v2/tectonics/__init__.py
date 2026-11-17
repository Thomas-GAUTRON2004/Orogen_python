"""Tectonic history model components ported from Orogen V2."""

from .crust import BeltSide, CrustFlag, CrustRaster, CrustType, OrogenyType, VolcanicType
from .config import CM_PER_YR_TO_KM_PER_MYR, CM_PER_YR_TO_RAD_PER_MYR
from .segments import SEGMENT_KM, fracture_coordinate
from .plate import Plate, frame_rotation
from .kinematics import (lat_lon_to_xyz, xyz_to_lat_lon, omega_from_euler_pole, surface_velocity, speed_cm_yr_at, advance_plates, pose_at_constant_omega)
from .occupancy import BinIndex, bin_index, mark_bin, may_have_crust_near
from .gather import clear_world_view, for_each_claim
from .polarity import overrides
from .reconstruct import reconstruct
from .contact import contact_at
from .resolve import resolve, candidate_cells, decide_birth
from .boundaries import Motion, classify_motion, vote_segments, near_land, feed_trench, classify_contact_boundaries
from .simulation import fastest_speed, closing_rate, choose_step
from .orogeny import belt_width_km, belt_shape, obliquity_deg
from .spread_belt import spread_belt
from .heights import belt_relief_km, broad_profile, build_heights
from .geology import thicken_cover
from .dynamics import solve3, drag_tensor, balanced_rotation, soften_speed, clamp_speed, update_plate_motion
from .forces import FORCE_KINDS, gather_boundary_loads
from .mantle import flow_of_cycle, mantle_flow_at, dispersal_field, repel_at
from .supercontinent import is_continental, landmasses, track_cycle
from .stepper import step_world
from .thermal import insulate_at_start, update_mantle_heat, plume_heat_factor, pick_plume_site
from .plumes import birth_plume, update_plumes
from .world import BoundaryKind, EventKind, Provisional, TectonicWorld

__all__ = ["BeltSide", "CrustFlag", "CrustRaster", "CrustType", "OrogenyType", "VolcanicType", "Plate", "frame_rotation", "BoundaryKind", "EventKind", "Provisional", "TectonicWorld", "BinIndex", "bin_index", "mark_bin", "may_have_crust_near", "clear_world_view", "for_each_claim", "overrides", "reconstruct", "contact_at", "resolve", "candidate_cells", "decide_birth", "Motion", "classify_motion", "vote_segments", "near_land", "feed_trench", "classify_contact_boundaries", "fastest_speed", "closing_rate", "choose_step", "belt_width_km", "belt_shape", "obliquity_deg", "spread_belt", "belt_relief_km", "broad_profile", "build_heights", "thicken_cover", "solve3", "drag_tensor", "balanced_rotation", "soften_speed", "clamp_speed", "update_plate_motion", "FORCE_KINDS", "gather_boundary_loads", "flow_of_cycle", "mantle_flow_at", "dispersal_field", "repel_at", "is_continental", "landmasses", "track_cycle", "step_world", "insulate_at_start", "update_mantle_heat", "plume_heat_factor", "pick_plume_site", "birth_plume", "update_plumes", "lat_lon_to_xyz", "xyz_to_lat_lon", "omega_from_euler_pole", "surface_velocity", "speed_cm_yr_at", "advance_plates", "pose_at_constant_omega", "CM_PER_YR_TO_KM_PER_MYR", "CM_PER_YR_TO_RAD_PER_MYR", "SEGMENT_KM", "fracture_coordinate"]
