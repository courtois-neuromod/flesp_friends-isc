#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=postproc
#SBATCH --output=./output_file_72.out
#SBATCH --error=./error_file_72.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make data USER_OPTIONS='--roi True'
