# Shared V2 configuration and data models

This document records the Python port boundary for issue #14. Each implementation is tied to its upstream v3.0.1 source at commit
`ed466f9db71292e912d8d0160e689a67a994f2d9`.

| Python API | Upstream JavaScript reference | Port contract |
|---|---|---|
| `orogen.v2.tectonics.config.TECTONICS` | `v2/js/tectonics/config.js` | Complete default configuration, including initial-world, boundaries, forces, belts, rifting, subduction, plumes, supercontinent, geology, far-field and reserved groups. |
| `MECHANISMS`, `mechanisms_off`, `only_mechanisms`, `mechanisms_on` | `v2/js/tectonics/config.js` | Named mechanism switches and deterministic override helpers. |
| `CrustRaster`, `CrustType`, `OrogenyType`, `VolcanicType`, `BeltSide`, `CrustFlag` | `v2/js/tectonics/model/crust.js` | Structure-of-arrays cell state, typed fields, count/mask invariants, create/copy/snapshot/restore/remove and history recording. |
| `Plate`, `frame_rotation` | `v2/js/tectonics/model/plate.js` | Rigid plate pose, frame transforms, cap/extent bookkeeping and frame-to-frame vector rotation. |
| `TectonicWorld`, `EventKind`, `BoundaryKind`, `Provisional` | `v2/js/tectonics/model/world.js` | World-frame ownership arrays, per-world isolated config, deterministic random streams, event log and unique terrane identifiers. |
| `GeoFlag`, `GEO_FRAME_FIELDS`, `GEOLOGY_READS`, encode/decode helpers | `v2/js/tectonics/model/geology.js` | Bit flags and byte/nibble quantization for the geology fields stored on crust and written into compact frames. |
| `SEGMENT_KM`, `fracture_coordinate` | `v2/js/tectonics/model/segments.js` | Fracture-zone coordinate measured in ridge-segment lengths. |

## Compatibility notes

- The upstream geology config imports `GEOLOGY.guideRules.coalByWetBelts`; the v3.0.1 source value is `true`, which is kept as the tectonic config default without introducing a dependency on the finish pipeline.
- JavaScript's `Math.round` behavior is preserved by explicit half-up rounding in the quantization helpers.
- With no explicit config, each world receives an isolated copy of the complete upstream defaults. An explicit config retains its supplied (possibly partial) shape; optional overrides are deep-copied and recursively merged into it. This is a deliberate behavior change from the former shallow `dict.update`: nested sibling keys are now preserved. Callers that relied on nested-group replacement should pass the complete intended group or adapt their override construction.
- Known fields are checked against the upstream schema (including nested group types and fixed-length parameter arrays); unknown keys are retained for forward compatibility. Mechanism switches are included in the validation schema, and positive infinity is accepted only where the upstream mechanism default itself is positive infinity.
- The Python API uses snake_case function names while retaining upstream camelCase data-field names where those names are part of the shared state contract.
- The geology encodings are lossy by design. Tests assert the upstream quantization behavior and sentinel values, not lossless floating-point round trips.

## Validation

- `tests/test_v2_tectonic_config.py`: upstream defaults, nested overrides, schema validation, and mechanism helpers.
- `tests/test_v2_tectonic_geology.py`: model flags, frame field contract, scalar encodings, flood-age sentinel and facies packing; it invokes the vendored upstream JavaScript oracle in `tests/reference/v2/geology-encoding.mjs` for exact encoded/decoded parity.
- `tests/reference/v2/tectonics/geology.mjs`: upstream v3.0.1 geology encoder implementation, retained unchanged apart from the test fixture's module boundary.
- Model audit trace: `tests/test_v2_tectonic_crust.py` ↔ upstream `v2/js/tectonics/model/crust.js`; `tests/test_v2_tectonic_plate.py` ↔ `plate.js`; `tests/test_v2_tectonic_world.py` and `tests/test_v2_differential_extended.py` ↔ `world.js`; `tests/test_v2_tectonic_segments.py` ↔ `segments.js`. These tests exercise field layout, typed-array dtypes, copy/snapshot/restore invariants, frame transforms, world arrays/RNG streams, and segment coordinates respectively.
