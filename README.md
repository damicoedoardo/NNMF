# NearestNeighborsMF

### Reproducibility
All the experiments presented in the paper are reproducible following the instructions:

#### Step 0
If you want to safely run the experiments in an Anaconda virtual environment, run the `create_env.sh` file in the project main folder and activate it with `source activate recsys-nnmf`.
Requires Anaconda3 to be installed.

#### Step 1
Run the `install.sh` file in the project main folder.

#### Step 2
Run the file `reproducibility1.py` located in the project main folder,
this file will train and save the optimal models necessary to reproduce the obtained results.

#### Step 3
Run the file `reproducibility2.py` located in the project main folder,
This file will let you select which experiment to rerun.
