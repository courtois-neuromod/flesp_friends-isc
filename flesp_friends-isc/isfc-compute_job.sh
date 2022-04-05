#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=isfc_friends
#SBATCH --output=./isfc_maps_10.out
#SBATCH --error=./isfc_maps_10.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make features USER_OPTIONS="--roi False --kind spatial"
