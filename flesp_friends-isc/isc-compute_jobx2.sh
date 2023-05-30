#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=ISC
#SBATCH --output=./isc_maps.out
#SBATCH --error=./isc_maps.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make more_features USER_OPTIONS="--roi False --kind temporal --pairwise False --slices False --lng 30"
