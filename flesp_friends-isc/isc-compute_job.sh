#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=postproc
#SBATCH --output=./isc_maps_6.out
#SBATCH --error=./isc_maps_6.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make features
