## Fourmi2024

This project aims to implement and parallelize an Ant Colony Optimization (ACO) algorithm to solve search and optimization problems. Here it is achived an acceleration of 2.02 by the paralization code. It was first implemented a simple paralization that separated the graphical render to the colony calculation and then one that would split the colonies for multiple cores.

Here are the main components of the project and how to execute them:

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

**Execution Instructions:**

mpiexec -n {number of processes} python3 Multicolonies_parallel.py
