# Detecting Anomalies with Autoencoders on Data Streams 

This repository contains the results for our ECML 2022 submission "Detecting Anomalies with Autoencoders on Data Streams"

## Installation
```shell
git clone git@github.com:LCa95/LCa95-DetectingAnomaliesWithAutoencodersOnDataStreams.git
```
```shell
virtualenv -p python3 online_anomaly_detection
```
```shell
source bin activate online_anomaly_detecion
```
```shell
cd DetectingAnomaliesWithAutoencodersOnDataStreams
```
```shell
pip install -r requirements.txt
```
## Reproducing the results
To run all experiments at once, run the `run_exps.sh` script located in `./scripts` by
```shell
./scripts/run_exps.sh
```
The evaluation results are stored within the `./evaluation` folder.

## Reproducing the results step by step
All Expereiments are located in `./tools`.
The evaluation results are stored within the `./evaluation` folder.

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



