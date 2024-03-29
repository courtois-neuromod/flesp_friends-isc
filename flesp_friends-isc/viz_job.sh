#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=viz-surf
#SBATCH --output=./viz_12.out
#SBATCH --error=./viz_12.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make figures USER_OPTIONS='--average subject --slices True --kind temporal --pairwise True'
