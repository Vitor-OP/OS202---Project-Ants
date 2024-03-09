## Fourmi2024

Ce projet vise à implémenter et à paralléliser un algorithme d'optimisation de colonie de fourmis (ACO) pour résoudre des problèmes de recherche et d'optimisation. Voici les principaux composants du projet et comment les exécuter :

### Principaux composants

1. **Code séquentiel:**  
   Le code séquentiel est une implémentation de base de l'algorithme ACO qui s'exécute de manière sérielle. Il est utile pour comprendre le fonctionnement de l'algorithme sans tenir compte de la parallélisation.

**Instructions d'exécution:**

cd Sequentiel/
python3 Sequential.py

2. **Code de parallélisation des tâches:**  
Le code Task_parallel.py implémente une version parallèle de l'algorithme ACO en utilisant la parallélisation des tâches. Il divise les tâches entre deux processus pour accélérer l'exécution.

**Instructions d'exécution :**

mpiexec -n {nº de processus} python3 Task_parallel.py

3. **Code de parallélisation de multicolonies:**  
Le code Multicolonies_parallel.py implémente une version parallèle de l'algorithme ACO en utilisant plusieurs colonies de fourmis qui fonctionnent simultanément. Cela peut améliorer la diversité de la recherche et la qualité de la solution

**Instructions d'exécution:**

mpiexec -n {nº de processus} python3 Multicolonies_parallel.py

### Exigences

- Python 3.x
- MPI (Message Passing Interface) pour Python (mpi4py)
