#!/bin/bash

echo 'Running assignment'

echo 'Training the model'
python3 train.py

echo 'Evaluating dev data'
python3 evaluate.py dev