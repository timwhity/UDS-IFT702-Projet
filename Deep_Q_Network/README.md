# Deep Q Network pour le jeu Demon Attack

Repris et modifié de J.Michaux : https://github.com/jmichaux/dqn-pytorch/blob/master/README.md 

Répertoire composé de :
-model weights : contient deux fichiers de poids différents 
-result : contient des fichiers csv des résultats des entrainements, pour différents changements d'hyperparamètres. 
Les caractéristiques sont les suivantes :
    - Le nombre d'épisode de jeu
    - Le nombre d'action total
    - Le nombre d'action sur une partie (on appelle ici un pas, mais il s'agit d'une des 6 actions possibles)
    - Le score obtenu 
    - La loss obtenue
- Un fichier graphs.ipynb qui permet d'afficher des graphes (reward et loss par rapport au nombre de pas total)
- le fichier main.ipynb qui permet de lancer les entrainements
- les fichiers test.py, memory.py et models.py (provient du répertoire initial, pas de modification)
- le fichier wrappers.py permettant d'apporter des modifications à l'environemment Gymnasium sans changer le code intitial. 

