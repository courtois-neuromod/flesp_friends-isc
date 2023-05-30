#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=LOO_Brain-ISC
#SBATCH --output=./output_files/iscs.out
#SBATCH --error=./error_files/iscs.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make features USER_OPTIONS="--roi False --kind temporal --pairwise True --slices True --lng 30"
