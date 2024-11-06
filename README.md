## Fourmi2024

This project aims to implement and parallelize an Ant Colony Optimization (ACO) algorithm to solve search and optimization problems. Here are the main components of the project and how to execute them:

### Main Components

**Work Report**

The file OS202_project_report.pdf contains a summary of the work developed, including the main analysis points and answers to the posed questions.
   
**Sequential Code:**

The sequential code is a basic implementation of the ACO algorithm that runs serially. It is useful for understanding the algorithm's functionality without considering parallelization.

**Execution Instructions:**

cd Sequentiel/
python3 Sequential.py

**Task Parallelization Code:**

The Task_parallel.py code implements a parallel version of the ACO algorithm using task parallelization. It splits tasks between two processes to speed up execution.

**Execution Instructions:**

mpiexec -n {number of processes} python3 Task_parallel.py

**Multicolonies Parallelization Code:**

The Multicolonies_parallel.py code implements a parallel version of the ACO algorithm using multiple ant colonies that operate simultaneously. This can improve search diversity and solution quality.

**Execution Instructions:**

mpiexec -n {number of processes} python3 Multicolonies_parallel.py
