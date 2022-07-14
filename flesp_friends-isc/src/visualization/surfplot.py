"""Publication ready surfplots"""
# general dependencies
import click
import glob
import itertools
# niimg dependencies
from neuromaps.transforms import mni152_to_fslr
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
from surfplot.utils import threshold
import nibabel as nib
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
# global vars
subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("--pairwise", type=bool)
@click.argument("--apply_threshold", type=float)
def surfplot(data_dir, pairwise=False, apply_threshold=None):
    """
    """
    if pairwise:
        pairs = []
        for pair in itertools.combinations(subjects, 2):
            pairs.append(pair[0] + "-" + pair[1])
        subjects = pairs
    for fname in subjects:
        # load stat map
        img = nib.load(f"{data_dir}/{fname}*.nii.gz")
        # convert niimg type
        gii_lh, gii_rh = mni152_to_fslr(img)
        # threshold raw temporal Brain-ISC
        if apply_threshold is None:
            data_lh = gii_lh.agg_data()
            data_rh = gii_rh.agg_data()
        else:
            data_lh = threshold(gii_lh.agg_data(), 0.2)
            data_rh = threshold(gii_rh.agg_data(), 0.2)

        # get surfaces + sulc maps
        surfaces = fetch_fslr()
        lh, rh = surfaces['inflated']
        sulc_lh, sulc_rh = surfaces['sulc']

        p = Plot(lh, rh, layout='row')
        p.add_layer({'left': sulc_lh, 'right': sulc_rh}, cmap='binary_r')

        # cold_hot is a common diverging colormap for neuroimaging
        cmap = nilearn_cmaps['cold_hot']
        p.add_layer({'left': data_lh, 'right': data_rh}, cmap=cmap,
                    color_range=(-11, 11))

        fig = p.build()
        fig.savefig()
