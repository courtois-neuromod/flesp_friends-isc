#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=second_level
#SBATCH --output=output_files/isc_hrmaps_sliced_12.out
#SBATCH --error=error_files/isc_hrmaps_sliced_12.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com

workon flesp_friends-isc
cd flesp_friends-isc
make model USER_OPTIONS='--seg_len 30 --pairwise True --roi False'
