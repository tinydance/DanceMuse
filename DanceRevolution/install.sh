#! /bin/bash

conda create --name music2dance python=3.7
conda activate music2dance
conda install cudatoolkit=10.0

pip install -r requirements.txt
