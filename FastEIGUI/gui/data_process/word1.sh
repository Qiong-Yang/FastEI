#!/bin/sh
#An example for gpu job.
#SBATCH -J spectrums_word
#SBATCH  -n10 -p cpuQ  --qos=cpuq 
source activate word
python spectrum_word.py
