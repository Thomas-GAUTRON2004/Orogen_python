# État du portage Orogen 3.0.1

Dernière mise à jour : 2026-11-16

## Cible

- Dépôt amont : https://github.com/raguilar011095/planet_heightmap_generation
- Version : `v3.0.1`
- Commit amont : `ed466f9db71292e912d8d0160e689a67a994f2d9`
- Référence principale : moteur V2 situé dans `v2/js/`

## État initial constaté

Le paquet Python historique disponible est en version 1.1.0. Il fournit déjà une API de génération de monde, de terrain, de géologie, de climat et d'environnement, mais son moteur d'évolution tectonique est une implémentation simplifiée et ne reproduit pas l'ensemble du moteur V2 JavaScript 3.0.1.

Le moteur V2 amont comprend environ 210 fichiers JavaScript et près de 46 000 lignes. Les principaux domaines identifiés sont :

| Domaine | Fichiers | Lignes approximatives |
|---|---:|---:|
| Tectonique | 63 | 10 239 |
| Terrain | 17 | 4 661 |
| Climat | 16 | 4 200 |
| Géologie | 8 | 2 227 |
| Finition | 13 | 3 998 |
| Agriculture | 4 | 854 |
| Core | 11 | 683 |
| Sauvegarde | 5 | 519 |

Ces chiffres servent à estimer le périmètre ; ils ne constituent pas une mesure de couverture Python.

## Plan d'exécution

Le portage est réalisé par lots fonctionnels, avec un commit GitHub par étape vérifiable :

1. **Socle Python** : structure du paquet, version, configuration, conventions numériques et reproductibilité des graines.
2. **Core et sauvegarde** : types partagés, sérialisation, chargement et validation des données.
3. **Tectonique** : initialisation des plaques, géométrie, frontières, vitesses, opérations et scénarios.
4. **Évolution** : moteur temporel, événements tectoniques, rotations et instantanés.
5. **Terrain** : relief, croûte, lithologie, rifts, dorsales et chaînes de montagnes.
6. **Géologie** : archives, collisions, subduction, volcanisme et dépôts.
7. **Climat** : circulation, températures, précipitations et biomes.
8. **Finition et agriculture** : hydrologie, érosion, glace, roches et systèmes agricoles.
9. **Parité** : fixtures déterministes, tests d'invariants et comparaisons numériques avec le moteur JavaScript (banc différentiel initial déjà ajouté pour seed/RNG, quaternions, raster crustal et plaques ; maillage et simulation complète restent à comparer).

L'ordre pourra être ajusté si les dépendances réelles entre modules l'exigent.

## Règles de validation

- Ne pas annoncer une fonctionnalité comme portée tant qu'elle n'a pas de tests Python.
- Distinguer les ports exacts, les adaptations numériques et les approximations.
- Comparer les sorties sur les mêmes graines, paramètres, maillages et instants.
- Documenter les tolérances numériques ; ne pas masquer les écarts par des seuils excessifs.
- Conserver des tests d'invariants même lorsqu'une comparaison exacte n'est pas possible.
- Ne déclarer la parité globale qu'après couverture des domaines fonctionnels et exécution des tests de référence.
- Maintenir une couverture de lignes de 100 % sur `orogen.v2`, contrôlée par le seuil bloquant de pytest-cov dans la CI.

## Blocages connus au démarrage

Les scripts de contrôle V2 amont dépendent de paquets Node.js, notamment `delaunator`. Les premières tentatives d'exécution locale ont échoué avant le lancement des contrôles faute de cette dépendance. Il faudra installer les dépendances amont déclarées et relancer les contrôles avant d'utiliser leurs résultats comme référence.

## Historique des lots

| Lot | État | Notes |
|---|---|---|
| Audit du périmètre amont | Terminé | Sources V2.3.0.1 inspectées ; domaines et volumes recensés. |
| Audit du paquet Python 1.1.0 | Terminé | Base existante identifiée ; parité complète non établie. |
| Socle Python 3.0.1 | En cours | Namespace de paquet, RNG/seed, bruit simplex, quaternions, géométrie sphérique, BFS, index sphérique, SimMesh, localisation barycentrique, tas min, Dijkstra, anneaux de voisinage, repère polaire d'Euler, classification des petits appareils, raster crustal complet, modèle de plaque rigide et conteneur d'état du monde (tableaux d'occupation, frontières, budgets, événements, RNG séparés) publiés. Le monde exige encore une configuration TECTONICS fournie par l'appelant. **107 tests réussis ; couverture de lignes de `orogen.v2` à 100 % (1 259/1 276 instructions).** Le seuil de 100 % est maintenant imposé par la CI. |
| Core V2 | En cours | Primitives numériques, bruit, rotations, géométrie sphérique, génération Fibonacci, projection stéréographique, fermeture du maillage au pôle Nord, triangulation Delaunay via SciPy/Qhull, index spatial, interpolation, graphe et opérations de voisinage. Le backend de production est SciPy/Qhull ; une comparaison avec Delaunator confirme les mêmes triangles et voisinages sur sept couples résolution/graine (12/99, 24/1, 48/17, 96/2026, 128/7, 256/42, 512/12345). Ce résultat reste borné à ces cas et ne garantit pas toutes les résolutions/graines. |
| Portage des domaines V2 | À faire | À publier par sous-systèmes. |
| Tests de parité | Partiel | Les tests différentiels passent contre les sources JS amont embarquées ; le maillage a une parité de connectivité sur sept cas. Les étapes dynamiques du TectonicWorld et les simulations multi-étapes restent à comparer. Aucune parité globale revendiquée. |
