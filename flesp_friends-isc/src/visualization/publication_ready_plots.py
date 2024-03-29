"""Publication ready surfplots"""
# general dependencies
import click
import glob
import itertools
import logging

# niimg dependencies
from neuromaps.transforms import mni152_to_fslr
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
from surfplot.utils import threshold
import nibabel as nib
from nilearn.image import mean_img


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("figures_dir", type=click.Path(exists=True))
@click.option("--pairwise", type=bool)
@click.option("--apply_threshold", type=float)
@click.option("--average", type=bool)
def surfplot(
    data_dir, figures_dir, pairwise=False, apply_threshold=None, average=False
):
    """
    Generate publication-ready surface plots.

    A function embedded callable through a command-line interface that generates
    publication-ready figures of statistical brain surface map (volume).

    Arguments
    ---------
    data_dir : path
        directory with stat maps (.nii.gz) of Brain-HR-ISC.
    figures_dir: path
        output directory (/scratch/data/figures).
    pairwise : bool
        whether of not to use the data from pairwise ISC method.
    apply threshold : float
        whether or not to apply a low-cut threshold to provided stat map.
    average : bool
        whether or not to average multiple stat maps (i.e. if using
        first order ISC volumes).

    Saves figure in a static image format showing both hemispheres
    """
    logger = logging.getLogger(__name__)
    # global vars
    subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
    if pairwise is True:
        pairs = []
        for pair in itertools.combinations(subjects, 2):
            pairs.append(pair[0] + "-" + pair[1])
        subjects = pairs
    for fname in subjects:
        logger.info(f"{fname}")
        if average is False:
            # load stat map
            img = nib.load(glob.glob(f"{data_dir}/{fname}*.nii.gz")[0])
        else:
            img = mean_img(glob.glob(f"{data_dir}/*.nii.gz"))
        # convert niimg type
        gii_lh, gii_rh = mni152_to_fslr(img)
        # threshold raw temporal Brain-ISC
        if apply_threshold is None:
            data_lh = gii_lh.agg_data()
            data_rh = gii_rh.agg_data()
        else:
            data_lh = threshold(gii_lh.agg_data(), apply_threshold)
            data_rh = threshold(gii_rh.agg_data(), apply_threshold)

        # get surfaces + sulc maps
        surfaces = fetch_fslr()
        lh, rh = surfaces["inflated"]
        sulc_lh, sulc_rh = surfaces["sulc"]
        logger.info("loaded surface")

        # constructing plot
        p = Plot(
            lh,
            rh,
            size=(800, 200),
            zoom=1.2,
            layout="row",
            mirror_views=True,
            brightness=0.8,
        )
        p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)

        # cold_hot is a common diverging colormap for neuroimaging
        p.add_layer(
            {"left": data_lh, "right": data_rh}, cbar_label="HR-ISC ~ Brain-ISC"
        )
        kws = dict(
            location="bottom",
            draw_border=False,
            aspect=10,
            shrink=0.2,
            decimals=0,
            pad=0,
        )
        fig = p.build(cbar_kws=kws)
        fig.axes[0].set_title(f"Brain-ISC regressed by HR-ISC \n {fname}", pad=-3)
        if average is True:
            fig.savefig(f"{figures_dir}/mean_HR-BrainISC.png", dpi=300)
            break
        else:
            fig.savefig(f"{figures_dir}/{fname}_HR-BrainISC.png", dpi=300)


if __name__ == "__main__":
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    surfplot()
