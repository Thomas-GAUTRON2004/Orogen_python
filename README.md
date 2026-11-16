# Orogen Python

Portage Python du projet Orogen (référence amont : [planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation), version 3.0.1).

## État publié

Cette branche initialise le dépôt et documente l'état du travail Python disponible. La base Python existante est la version **1.1.0** ; elle ne constitue **pas** encore un portage complet et équivalent de la version JavaScript 3.0.1.

Le code Python 1.1.0 couvre une partie des fonctionnalités de génération de monde, de terrain, de climat et d'environnement. Le moteur V2 d'évolution tectonique et plusieurs systèmes avancés de la référence 3.0.1 restent à porter et à valider.

## Référence amont

- Dépôt : https://github.com/raguilar011095/planet_heightmap_generation
- Version ciblée : `v3.0.1`
- Commit amont : `ed466f9db71292e912d8d0160e689a67a994f2d9`

## Validation

Aucune parité complète avec le moteur JavaScript 3.0.1 n'est revendiquée à ce stade. Les scripts de vérification V2 nécessitent notamment la dépendance Node `delaunator`.

## Suite du portage

1. Publier les sources Python 1.1.0 existantes.
2. Porter les domaines V2 (tectonique, terrain, climat, géologie, finition, agriculture).
3. Ajouter des tests de non-régression comparant les sorties Python et JavaScript.
4. Ne déclarer la parité atteinte qu'après exécution des tests comparatifs.
