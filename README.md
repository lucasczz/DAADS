# Detecting Anaomalies with Autoencoders on Data Streams 

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
To run all experiments at once, navigate to `./scripts` by 
```shell 
cd scripts
```
and run 
```shell
run_exps.sh
```

## Reproducing the results step by step
All Expereiments are located in `./tools`. To run all experiments navigate to the `tools` folder.
The evaluation results are stored within the `./evaluation` folder.
### Run Benchmarks
```shell
python benchmark_exp.py
```
### Run Benchmark experiments
```shell
python benchmark_exp.py
```
### Run Contamination experiment
```shell
python contamination_exp.py
```
### Run Capacity experiment
```shell
python capacity_exp.py
```
### Run learning rate experiment
```shell
python lr_exp.py
```




