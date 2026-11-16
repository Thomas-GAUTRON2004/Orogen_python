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
python -m pytest
```

Dépendances de calcul actuelles : NumPy et SciPy (Qhull pour la triangulation Delaunay). Le backend Delaunay est différent de Delaunator en JavaScript : la topologie et les invariants sont testés, mais une comparaison différentielle sur des graines communes reste nécessaire avant de revendiquer une identité des maillages. Les dépendances des futurs sous-systèmes seront
ajoutées au fur et à mesure de leur portage.

## Contenu actuellement porté

Le namespace `orogen.v2.core` comprend les premières traductions des briques V2 :

- génération déterministe des graines et RNG Park–Miller ;
- bruit simplex 3D et FBM ;
- rotations par quaternions ;\n- repère polaire autour d’un pôle d’Euler et classification du budget mémoire des petits appareils ;
- géométrie sphérique ;
- génération Fibonacci, projection stéréographique et fermeture du maillage au pôle Nord ;
- triangulation Delaunay via SciPy/Qhull et construction du maillage de simulation ;
- index spatial sur la sphère et reconstruction d'un maillage depuis ses tableaux ;
- parcours BFS, distances en nombre de sauts, composants connexes ;
- localisation barycentrique sur un triangle sphérique ;
- tas min et propagation Dijkstra en kilomètres ;
- pré-calcul des anneaux de voisinage.

Les fonctions sont encore des primitives internes. Le backend Delaunay est SciPy/Qhull
et non Delaunator ; les invariants topologiques sont testés, mais l'identité exacte des
connectivités doit encore être comparée avec JavaScript. Le moteur tectonique temporel
et les systèmes de terrain, géologie, climat, finition et agriculture restent à porter.

## Référence et provenance

- Version amont : `v3.0.1`
- Commit amont : `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Licence : GPL-3.0-only (voir `LICENSE` et `NOTICE.md`).

## Tests et parité

Les fixtures déterministes de RNG et de bruit sont calculées à partir du moteur JavaScript
amont. Les tests d'invariants couvrent les rotations, le maillage et les opérations de graphe.

La CI GitHub Actions exécute `pytest` sur les branches de portage et les pull requests.
La dernière exécution vérifiée a réussi : **25 tests passés** sur Python 3.12, incluant la topologie sphérique, la cyclicité des voisins et la localisation sur le maillage généré.

La parité globale ne sera déclarée qu'après comparaison des sorties Python et JavaScript
sur les mêmes graines, paramètres, maillages et instants.
