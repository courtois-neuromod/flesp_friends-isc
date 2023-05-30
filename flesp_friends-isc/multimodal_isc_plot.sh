#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=Viz_multimodal-isc_mosaicplots
#SBATCH --output=./output_files/multimodalISC_plot.out
#SBATCH --error=./error_files/multimodalISC_plot.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s5c7o8a6s1u8i7u8@courtois-neuromod.slack.com
workon flesp_friends-isc
cd flesp_friends-isc
python3 src/visualization/visualize.py /scratch/flesp/data/second_level_models/pw_segments_threshold0.2_30TRs/ /scrat
ch/flesp/ --average subject --kind multimodal --pairwise True --slices False
