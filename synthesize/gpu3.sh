#!/bin/sh
#SBATCH -J 1(AC)
#SBATCH -N 1
#SBATCH -p gpu03
#SBATCH --gres=gpu:1
#SBATCH --ntasks=32

python main_2.py