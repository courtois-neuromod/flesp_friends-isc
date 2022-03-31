#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=postproc
#SBATCH --output=./output_file.out
#SBATCH --error=./error_file.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make data
