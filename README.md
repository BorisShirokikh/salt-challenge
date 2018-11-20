# salt-challenge
Mini-lib to solve TGS Salt Identification Challenge in different ways

## Jupyter Notebook
1. To run jupyter on server you first need to activate environment, e.g. `source miniconda3/bin/activate`, then start jupyter:

```
jupyter-notebook --no-browser --port=PORT_ID
```

2. To bridge your notebook from server use `ssh`:

```
ssh -L localhost:PORT_ID:localhost:PORT_ID -N hostname
```

## Run experiments

Command to run already built experiment:

```
bash /path/to/salt-challenge/scripts/do_sequence.sh EXPERIMENT_PATH /path/to/salt-challenge/scripts
```
