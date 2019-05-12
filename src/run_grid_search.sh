#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH -J WikiLong
#SBATCH --gres=gpu:8

python grid_search.py