# Adaptations Python par rapport à l'amont JavaScript

Référence : `planet_heightmap_generation` @ `ed466f9db71292e912d8d0160e689a67a994f2d9` (tag v3.0.1), submodule `third_party/planet_heightmap_generation`.

Ce document distingue les **ports directs** (même algorithme, même ordre de calcul, vérifiés par parité) des **adaptations Python** (écarts assumés ou à décider). Chaque adaptation indique la preuve mesurée contre le JavaScript.

## 1. Ports directs (vérifiés)

| Élément | Preuve |
|---|---|
| `TECTONICS` : 310 clés et valeurs par défaut | `tests/test_v2_tectonic_config_upstream.py` (fixture généré depuis le submodule) |
| Bruit `SimplexNoise` (12 gradients, fBm, fBm en crête) | 14 400 comparaisons bit à bit (`tests/test_v2_simplex_noise.py`) |
| Monde initial : 6 configurations (512, 768, 1 024 cellules ; six graines) | `tests/test_v2_tectonic_initial_world.py::test_production_resolution_start_world_matches_javascript_reference` |
| 37 scènes : identifiants, libellés, durée, vue, texte | Métadonnées identiques au JS |
| 32 scènes (`build`) : plaques, cellules, types de croûte | `tests/test_v2_scene_parity.py`, quatre configurations mesurées |
| Lettres des cratons | Identiques à `String.fromCharCode(65 + i)` |
| Sémantique `%` (signe du dividende) | `math.fmod` aux sites portés |
| Distances `spreadKm` : stockage float32 à chaque relaxation | Écart d'un ulp corrigé, parité du monde initial |
| Constructeur `TectonicWorld(mesh, seed, overrides=None)` | Même signature et même fusion que `new TectonicWorld(simMesh, seed, overrides)` |

## 2. Adaptations Python : validation de configuration

Le JavaScript n'a **aucun** validateur. Chaque règle Python ci-dessous est un ajout. Mesures faites avec `buildStartWorld` sur 768 cellules, graine 941.

| Règle Python | Comportement JS mesuré | Décision |
|---|---|---|
| `startArcKm` : largeurs strictement positives | `[0, 0]` : crash mémoire (abort, rc 134) | **Conservée** : justifiée |
| `startMaxLatitudeDeg` strictement positif, ≤ 90 | `0` : boucle infinie (timeout) | **Conservée** : justifiée |
| `coastUnionSoftnessKm` strictement positif | `0` : renvoie un résultat dégradé (`fractionMet: false`, 24 tentatives), sans erreur | **À décider** : le Python rejette ; une division par zéro lèverait une exception Python. Option : émuler l'arithmétique IEEE |
| `startGrowthBias` < 1 | `1.0` : accepté, résultat normal | **Supprimée** |
| Autres règles `_POSITIVE`, `_NON_NEGATIVE`, `_UNIT_INTERVAL` | Non mesuré | **À vérifier** contre le JS avant de les conserver |

Les tests de limites (`tests/test_v2_tectonic_config_limits.py`, 290 cas) fixent le contrat Python actuel. Ils ne prouvent pas l'accord avec le JS pour les règles marquées « à vérifier ».

## 3. Adaptations Python : construction et erreurs

| Élément | Comportement JS | Python | Statut |
|---|---|---|---|
| `fractionMet` non atteint | Renvoie le layout le plus proche, `fractionMet: false` | Idem | Aligné |
| Aucun candidat | Lève `These settings do not fit on the planet for seed …` | Même texte | Aligné |
| Base de config partielle | Inexistante | Supprimée (`overrides` seul) | Aligné |

## 4. Adaptations Python : performance (sans effet sémantique)

- `SimplexNoise` est en Python pur : environ 10 µs par appel fBm. Les noyaux Numba existants (`orogen/v2/core/_numba_kernels.py`) accélèrent **FastNoise**, pas `SimplexNoise`. Les chemins du monde initial et des scènes ont donc perdu l'accélération Numba. Un noyau `SimplexNoise` bit-exact est une étape de performance à mesurer.
- `_outline_gap` (recherche de disposition) est O(72²) dans le pire cas. Construction du monde initial : 7,3 s (768 cellules), 7,8 s (1 024), 13 s (4 096).

## 5. Fixtures de test sans équivalent JS

Les builders `build_continental_scene`, `build_oceanic_scene`, `build_mixed_scene` et le dispatcher `build_scenario_scene` sont des **fixtures de test** : ils vivent dans `tests/scenario_fixtures.py` et ne font pas partie de l'API du paquet. Aucun équivalent upstream n'existe (recherche sur tout le submodule). Ils ne peuvent donc pas être comparés au JavaScript. Ils sont testés pour leurs contrats (couverture, invariants, scénario reporté).

## 6. Audit des modèles de données partagés

Modules upstream de `v2/js/core/`, `v2/js/tectonics/model/` et `v2/js/tectonics/config.js` : 17 modules. Tous ont une contrepartie Python :

| Upstream | Python |
|---|---|
| `core/device.js` | `orogen/v2/core/device.py` |
| `core/fast-noise.js` | `orogen/v2/core/fast_noise.py` |
| `core/mesh-bfs.js` | `orogen/v2/core/mesh_bfs.py` |
| `core/mesh-locate.js` | `orogen/v2/core/mesh_locate.py` |
| `core/mesh-rings.js` | `orogen/v2/core/mesh_rings.py` |
| `core/min-heap.js` | `orogen/v2/core/min_heap.py` |
| `core/rotation.js` | `orogen/v2/core/rotation.py` |
| `core/seed.js` | `orogen/v2/core/seed.py` |
| `core/sim-mesh.js` | `orogen/v2/core/sim_mesh.py` |
| `core/sphere-geom.js` | `orogen/v2/core/sphere_geometry.py` |
| `core/sphere-index.js` | `orogen/v2/core/sphere_index.py` |
| `tectonics/config.js` | `orogen/v2/tectonics/config.py` |
| `tectonics/model/crust.js` | `orogen/v2/tectonics/crust.py` |
| `tectonics/model/geology.js` | `orogen/v2/tectonics/geology.py` |
| `tectonics/model/plate.js` | `orogen/v2/tectonics/plate.py` |
| `tectonics/model/segments.js` | `orogen/v2/tectonics/segments.py` |
| `tectonics/model/world.js` | `orogen/v2/tectonics/world.py` |

Présence confirmée. Parité **exacte** : voir section 1 pour ce qui est mesuré.

## 7. Non vérifié

- Autres modèles de données hors de ce périmètre (par exemple `present/`, les snapshots) : non audités.
- Règles de validation marquées « à vérifier » en section 2.
