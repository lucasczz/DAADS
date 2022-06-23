# Detecting Anomalies with Autoencoders on Data Streams 

This repository contains the results for our ECML 2022 submission "Detecting Anomalies with Autoencoders on Data Streams"

## Installation
```shell
git clone https://github.com/lucasczz/DAADS.git
```
```shell
python3 -m venv daads_env
```
```shell
source daads_env/bin/activate
```
```shell
cd DAADS
```
```shell
pip install -r requirements.txt
```
## Reproducing the results
To run all experiments at once, run the `run_exps.sh` script located in `./scripts` by
```shell
./scripts/run_exps.sh
```
The experiment results are stored in `./results`.

## Reproducing the results step by step
All experiment scripts are located in `./tools`.

### Evaluate all models
```shell
python ./tools/benchmark_exp.py
```
### Run contamination experiment
```shell
python ./tools/contamination_exp.py
```
### Run capacity experiment
```shell
python ./tools/capacity_exp.py
```
### Run learning rate experiment
```shell
python ./tools/lr_exp.py
```
### Obtain anomaly scores
```shell
python ./tools/scores_exp.py
```

## Access datasets 
```shell 
from IncrementalTorch.datasets import Covertype, Shuttle
from river.datasets import CreditCard
```


