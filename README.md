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
# Node.js 22+ et Delaunator sont requis pour les tests différentiels.
npm install --no-save --package-lock=false delaunator@5.0.1
python -m pytest --cov=orogen.v2 --cov-report=term-missing
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
- le modèle de plaque rigide, transformations de repère, calotte de culling et rotations entre plaques ;
- la cinématique (conversion Euler/vitesse, rotation temporelle des plaques) et l'index spatial d'occupation ;
- la collecte des revendications, la reconstruction en lecture seule des propriétaires et les règles de polarité de subduction.

Les fonctions sont encore des primitives internes. Le backend de production Delaunay reste SciPy/Qhull, tandis que le banc de référence exécute Delaunator. Les connectivités, voisinages, anneaux BFS, distances en sauts, zones atteignables et propagations Dijkstra sont comparés sur sept maillages générés. Les connectivités sont identiques sur sept jeux de test différentiels (12, 24, 48, 96, 128, 256 et 512 points, avec des graines distinctes), mais cette comparaison ne constitue pas une preuve pour toutes les graines et résolutions. Les modèles de plaque et de raster crustal sont maintenant portés, mais les valeurs par défaut de configuration TECTONICS, l'initialisation des mondes, le résolveur, les étapes de tectonique temporelle et les systèmes de terrain, géologie, climat, finition et agriculture restent à porter.

## Référence et provenance

- Version amont : `v3.0.1`
- Commit amont : `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Licence : GPL-3.0-only (voir `LICENSE` et `NOTICE.md`).

## Tests et parité

Le banc différentiel exécute les sources JavaScript amont embarquées depuis le tag `v3.0.1`. Il compare notamment :
- parsing des graines, RNG Park–Miller et bruit simplex/FBM ;
- quaternions, géométrie sphérique et repère polaire ;
- génération Fibonacci, projection, topologie Delaunay et voisinage sphérique sur sept tailles/graines ;
- index cube-map, reconstruction depuis les tableaux, localisation barycentrique, BFS, anneaux, tas min et Dijkstra ;
- tous les champs du raster crustal, copie/transfert/restauration, plaques, transformations et état initial du monde tectonique avec ses flux RNG.

La CI GitHub Actions installe Node.js 22, Delaunator 5.0.1 et Python 3.12. Dernière CI vérifiée : **89 tests passés**. La couverture mesurée sur `orogen.v2` est de **96 %** (1 259 instructions, 53 non couvertes), avec un rapport `coverage.xml` produit par la CI. Les tests différentiels JavaScript/Python sont inclus dans ces 89 tests.

La parité globale n'est pas établie : les étapes tectoniques dynamiques, la simulation multi-pas et les systèmes terrain, géologie, climat, finition et agriculture ne sont pas encore comparés aux sorties JavaScript. Les sept cas de topologie Delaunay réussis ne couvrent pas toutes les graines ni toutes les résolutions.
