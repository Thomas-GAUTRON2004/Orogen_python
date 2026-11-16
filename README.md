# Orogen Python

Portage progressif en Python du moteur Orogen **3.0.1**, à partir de la référence amont :
[planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation).

> **État de développement :** le portage complet n'est pas terminé. La branche `port/v3.0.1`
> contient le socle numérique V2 et ses premiers tests ; elle ne revendique pas encore
> l'équivalence fonctionnelle avec le moteur JavaScript.

## Version et installation

Version de développement : `3.0.1.dev0`. Python 3.10 ou supérieur.

```bash
python -m pip install -e ".[test]"
# Node.js 22+ est requis pour les tests différentiels avec le moteur JavaScript original.
python -m pytest
```

Dépendances de calcul actuelles : NumPy et SciPy (Qhull pour la triangulation Delaunay). Le backend Delaunay est différent de Delaunator en JavaScript : la topologie et les invariants sont testés, mais une comparaison différentielle sur des graines communes reste nécessaire avant de revendiquer une identité des maillages. Les dépendances des futurs sous-systèmes seront
ajoutées au fur et à mesure de leur portage.

## Contenu actuellement porté

Le namespace `orogen.v2.core` comprend les premières traductions des briques V2 :

- génération déterministe des graines et RNG Park–Miller ;
- bruit simplex 3D et FBM ;
- rotations par quaternions ;
- repère polaire autour d’un pôle d’Euler et classification du budget mémoire des petits appareils ;
- géométrie sphérique ;
- génération Fibonacci, projection stéréographique et fermeture du maillage au pôle Nord ;
- triangulation Delaunay via SciPy/Qhull et construction du maillage de simulation ;
- index spatial sur la sphère et reconstruction d'un maillage depuis ses tableaux ;
- parcours BFS, distances en nombre de sauts, composants connexes ;
- localisation barycentrique sur un triangle sphérique ;
- tas min et propagation Dijkstra en kilomètres ;
- pré-calcul des anneaux de voisinage ;

Le namespace `orogen.v2.tectonics` comprend maintenant :
- les constantes et drapeaux de croûte, orogenèse, volcanisme et côtés de ceintures ;
- le raster de croûte par plaque avec ses champs typés, snapshots/restauration, transferts, subduction et enregistrement des orogenèses ;
- le modèle de plaque rigide, rotations de repère, calotte de culling et calcul de la rotation entre plaques.

Les fonctions sont encore des primitives internes. Le backend Delaunay est SciPy/Qhull
et non Delaunator ; les invariants topologiques sont testés, mais l'identité exacte des
connectivités doit encore être comparée avec JavaScript. Les modèles de plaque et de raster crustal sont maintenant portés, mais les valeurs par défaut de configuration TECTONICS, l'initialisation des mondes, le résolveur, les étapes de tectonique temporelle et les systèmes de terrain, géologie, climat, finition et agriculture restent à porter.

## Référence et provenance

- Version amont : `v3.0.1`
- Commit amont : `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Licence : GPL-3.0-only (voir `LICENSE` et `NOTICE.md`).

## Tests et parité

Un banc de tests différentiels exécute les modules JavaScript originaux embarqués depuis le tag amont `v3.0.1` et compare leurs sorties Python/JavaScript sur les mêmes entrées : parsing des graines, RNG Park–Miller, opérations quaternioniques, champs du raster crustal, transformations de plaques et calcul de calotte. Les sources de référence sont conservées sans modification algorithmique dans `tests/reference/v2/` ; seules les extensions et les chemins d'import ont été adaptés au rangement de test.

La CI GitHub Actions installe Node.js 22 et Python 3.12, puis exécute l'ensemble des tests. Dernière CI vérifiée : **35 tests passés**, dont 4 tests différentiels avec l'original JavaScript.

La parité globale n'est pas établie : le maillage Delaunay, le monde tectonique complet, le moteur d'évolution et les autres sous-systèmes ne sont pas encore couverts par des comparaisons différentielles.
