#!/bin/bash
python tools/benchmark_exp.py | python tools/lr_exp.py | python tools/contam_exp.py | python tools/capacity_exp.py | python tools/scores_exp.py | python tools/hst_exp.py
