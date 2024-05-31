# Bayesian Optimization of Functions over Node Subsets in Graphs

## Overall Structure - Demo
<img width="2237" alt="image" src="https://github.com/LeonResearch/GraphComBO/assets/64602721/b1f56340-278a-45b1-865c-77c30139c183">


### Search Animation 
<div align="center">
  <img src="https://github.com/LeonResearch/GraphComBO/assets/64602721/f8c13e14-53b4-470c-ae03-03932b7eedeb" alt="combo-subgraph" style="width: 60%; height: 60%">
</div>

## Recursive Combo-subgraph Sampling - Demo

<div align="center">
  <img src="https://github.com/LeonResearch/GraphComBO/assets/64602721/ee5fe868-4740-4af3-86f5-26101329492d" alt="combo-subgraph" style="width: 60%; height: 60%">
</div>

## Create virtual env & install dependencies
```
conda create -n graph
conda install networkx numpy pandas matplotlib seaborn scipy jupyterlab
conda install pyg -c pyg
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda activate graph
pip install ndlib
pip install osmnx
```

## Run
Use the following code in a bash shell to run an experiment with pre-specified configurations:
```bash
python main.py --problem BA
python main.py --problem WS
python main.py --problem GRID
python main.py --problem SBM
python main.py --problem Transitivity
python main.py --problem GNN_attack
python main.py --problem Coauthor_IC
python main.py --problem SIR
python main.py --problem Patient_Zero
```
