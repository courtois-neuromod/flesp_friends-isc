#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=Sliced_ISC
#SBATCH --output=./isc_maps_sliced_10.out
#SBATCH --error=./isc_maps_sliced_10.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make features USER_OPTIONS="--roi False --kind temporal --pairwise True --slices True --lng 30"
