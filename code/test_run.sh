#!/bin/bash
# This script is used to run the test suite for the project.
python main.py --exp_name test1 --task train_all --arch lstm --num_epochs 2 --dataset nih --seed 42
