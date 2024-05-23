# Bayesian Optimization of Functions over Node Combinations in Graphs
This is the python implementation with BOTorch for the paper Bayesian Optimization of Functions over Node Combinations in Graphs.

## Create virtual env & install dependencies
```
conda create -n graph
conda install networkx numpy pandas matplotlib seaborn jupyterlab
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda activate graph
pip install ndlib
```

## Run
Use the following code in a bash shell to run an experiment with pre-specified configurations:
```bash
python main.py --problem BA --plot_result True
```
