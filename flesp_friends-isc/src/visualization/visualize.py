"""Viz."""
import glob
import os
import click
import logging
import itertools
from nilearn import image, plotting, surface, maskers
from nilearn.datasets import fetch_surf_fsaverage
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import fnmatch

fsaverage = fetch_surf_fsaverage(mesh="fsaverage")
mask_name = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
brain_mask = nib.load(mask_name)
subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
episodes = glob.glob("/scratch/flesp/data/pw_isc-segments30/*/")
tasks = []
for task in sorted(episodes):
    tasks.append(task[-13:-1])

def _list_averaging_everything(data_dir, kind):
    """
    list averaged isc volumes
    """
    logger = logging.getLogger(__name__)
    logger.info("averaging all tasks from all subs")
    isc_files = sorted(glob.glob(f"{data_dir}/*/*.nii.gz"))
    isc_files = fnmatch.filter(isc_files, f"*{kind}*")
    isc_volumes = [image.mean_img(isc_files, verbose=5, n_jobs=-1)]
    logger.info("Averaged BOLD images")

    return isc_volumes
def _list_averaging_subjectwise(data_dir, subject, kind):
    """
    list averaged isc volumes
    """
    logger = logging.getLogger(__name__)
    logger.info(f"averaging all tasks from {subject}")
    isc_files = sorted(glob.glob(f"{data_dir}/*/{subject}*.nii.gz")) 
    #isc_files = fnmatch.filter(isc_files, f"*{kind}*")
    isc_volumes = [image.mean_img(isc_files, verbose=5, n_jobs=-1)]
    logger.info("Averaged BOLD images")

    return isc_volumes


def _list_averaging_taskwise(data_dir, tasks, subject, kind, slices):
    """
    list averaged isc volumes
    """
    logger = logging.getLogger(__name__)
    # iterate for a given sub
    for task in tasks:
        isc_volumes = []
        logger.info(f"{subject} | {task} ")
        isc_files = sorted(glob.glob(f"{data_dir}/{task}/{subject}*.nii.gz"))
        isc_files = fnmatch.filter(isc_files, f"*{kind}*")
        try:
            if slices is True:
                for fn in isc_files:
                    isc_volumes.append(image.mean_img(fn))
            else:
                isc_volumes = [image.mean_img(isc_files)]
        except StopIteration:
            logger.info("No ISC map for this episode segment")
            continue

        logger.info("Averaged BOLD images")
    return isc_volumes


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("figures_dir", type=click.Path(exists=True))
@click.option("--kind", type=str)
@click.option("--average", type=str)
@click.option("--slices", type=bool)
@click.option("--pairwise", type=bool)
def mosaic_surface_isc_plots(
    data_dir,
    figures_dir,
    subjects=subjects,
    tasks=tasks,
    kind="temporal",
    views=["lateral", "medial"],
    hemi=["left", "right"],
    threshold=0.15,
    vmax=1.0,
    average="subject",
    slices=False,
    pairwise=False,
):
    """
    Plot surface subject-wise.

    Parameters
    ----------
    subject : str
        Subject identifier for which to generate ISC plots.
    task: list
        Tasks for which to generate ISC plots.
    views : list
        View for which to generate ISC plots. Accepted values are
        ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'].
        Defaults to ['lateral', 'medial'].
    """
    logger = logging.getLogger(__name__)
    if pairwise is True:
        pairs = []
        for pair in itertools.combinations(subjects, 2):
            pairs.append(pair[0] + "-" + pair[1])
        subjects = pairs
    for subject in subjects:
        # list isc volumes by subject or subject/task
        if average == "task":
            isc_volumes = _list_averaging_taskwise(
                data_dir, tasks, subject, kind, slices
            )
        elif average == "all":
            isc_volumes = _list_averaging_everything(
                data_dir, kind
            )
        else:
            isc_volumes = _list_averaging_subjectwise(data_dir, subject, kind)
        # plot left hemisphere
        for idx, average_isc in enumerate(isc_volumes):
            if slices is True:
                fig_title = f"{subject} {task} segment {idx:02d}"
            elif average == "task":
                fig_title = f"{subject} {task}"
                fn = str(
                    f"{figures_dir}/{task}/"
                    f"mosaic_surfplot_{kind}"
                    f"ISC_on_{task}seg{idx:02d}_{subject}.png"
                )
            elif average == "subject":
                fig_title = ""
                fn = str(
                    f"{figures_dir}/"
                    f"mosaic_surfplot_{kind}"
                    f"ISC_on_all_{subject}.png"
                )
            else:
                fig_title = ""
                fn = str(
                    f"{figures_dir}/"
                    f"mosaic_surfplot_{kind}"
                    f"ISC_on_all_every-sub.png"
                )
            plotting.plot_img_on_surf(
                average_isc,
                fsaverage,
                title=fig_title,
                views=views,
                hemispheres=hemi,
                vmax=5,
                threshold=threshold,
                cmap="magma",
            )
            if os.path.exists(fn):
                os.remove(fn)
            try:
                plt.savefig(fn, bbox_inches="tight", dpi=300)
            except FileNotFoundError:
                logger.info(f"Creating path for {task}")
                os.mkdir(f"{figures_dir}/{task}/")
                plt.savefig(fn, bbox_inches="tight", dpi=300)
            plt.close("all")
            if average == "all":
                logger.INFO("Finished saving figure for all")
                break


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("figures_dir", type=click.Path(exists=True))
@click.option("--kind", type=str)
@click.option("--average", type=str)
@click.option("--slices", type=bool)
@click.option("--pairwise", type=bool)
def surface_isc_plots(
    data_dir,
    figures_dir,
    subjects=subjects,
    tasks=tasks,
    kind="temporal",
    views=["lateral", "medial"],
    hemi=["left", "right"],
    threshold=0.15,
    vmax=1.0,
    average="subject",
    slices=False,
    pairwise=False,
):
    """
    Plot surface subject-wise.

    Parameters
    ----------
    subject : str
        Subject identifier for which to generate ISC plots.
    task: list
        Tasks for which to generate ISC plots.
    views : list
        View for which to generate ISC plots. Accepted values are
        ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'].
        Defaults to ['lateral', 'medial'].
    """
    logger = logging.getLogger(__name__)
    if pairwise:
        pairs = []
        for pair in itertools.combinations(subjects, 2):
            pairs.append(pair[0] + "-" + pair[1])
        subjects = pairs
    for subject, view in itertools.product(subjects, views):
        # list isc volumes by subject or subject/task
        if average != "subject":
            isc_volumes = _list_averaging_taskwise(
                data_dir, tasks, subject, kind, slices
            )
        else:
            isc_volumes = _list_averaging_subjectwise(data_dir, subject, kind)
        # plot left hemisphere
        for idx, average_isc in enumerate(isc_volumes):
            if slices:
                fig_title = f"{subject} {task} segment {idx:02d}"
            elif average != "subject":
                fig_title = f"{subject} {task}"
                fn_left = str(
                    f"{figures_dir}/{task}/"
                    f"left_{view}_surfplot_{kind}"
                    f"ISC_on_{task}seg{idx:02d}_{subject}.png"
                )
                fn_right = str(
                    f"{figures_dir}/{task}/"
                    f"right_{view}_surfplot_{kind}"
                    f"ISC_on_{task}seg{idx:02d}_{subject}.png"
                )
            else:
                fig_title = ""
                fn_left = str(
                    f"{figures_dir}/"
                    f"left_{view}_surfplot_{kind}"
                    f"ISC_on_all_{subject}.png"
                )
                fn_right = str(
                    f"{figures_dir}/"
                    f"right_{view}_surfplot_{kind}"
                    f"ISC_on_all_{subject}.png"
                )
            texture = surface.vol_to_surf(average_isc, fsaverage.pial_left)
            plotting.plot_surf_stat_map(
                fsaverage.pial_left,
                texture,
                hemi=hemi,
                colorbar=True,
                cmap="magma",
                threshold=threshold,
                vmax=vmax,
                bg_map=fsaverage.sulc_left,
                view=view,
                title=fig_title,
            )
            if os.path.exists(fn_left):
                os.remove(fn_left)
            try:
                plt.savefig(fn_left, bbox_inches="tight", dpi=300)
            except FileNotFoundError:
                logger.info(f"Creating path for {task}")
                os.mkdir(f"{figures_dir}/{task}/")
                plt.savefig(fn_left, bbox_inches="tight", dpi=300)

            # plot right hemisphere
            texture = surface.vol_to_surf(average_isc, fsaverage.pial_right)
            plotting.plot_surf_stat_map(
                fsaverage.pial_right,
                texture,
                hemi=hemi,
                colorbar=True,
                cmap="magma",
                threshold=threshold,
                vmax=vmax,
                bg_map=fsaverage.sulc_right,
                view=view,
                title=fig_title,
            )
            plt.savefig(fn_right, bbox_inches="tight", dpi=300)
            del texture
            plt.close("all")


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
def plot_corr_mtx(data_dir, mask_img=brain_mask, kind="temporal"):
    """
    Plot Correlation matrix.

    Description.
    Parameters
    ----------
    kind : str
        Kind of ISC, must be in ['spatial', 'temporal']
    data_dir : str
        The path to the postprocess data directory on disk.
        Should contain all generated ISC maps.
    mask_img : str
        Path to the mask image on disk.
    """
    logger = logging.getLogger(__name__)
    from netneurotools.plotting import plot_mod_heatmap

    logger.info("Imported netneurotools util")
    if kind not in ["spatial", "temporal"]:
        err_msg = "Unrecognized ISC type! Must be spatial or temporal"
        raise ValueError(err_msg)

    logger.info(f"Season 1 communities")
    isc_files = sorted(glob.glob(f"{data_dir}/*s01*/*.nii.gz"))
    masker = maskers.NiftiMasker(mask_img=mask_img)
    logger.info("Mask loaded")

    isc = [masker.fit_transform(i).mean(axis=0) for i in isc_files]
    logger.info("Mask applied")
    corr = np.corrcoef(np.row_stack(isc))
    logger.info("Computed cross-correlation")

    # our 'communities' are which film was presented
    episodes = [i.split("task-s01")[-1].strip("_temporalISC.nii.gz") for i in isc_files]
    num = [i for i, m in enumerate(set(episodes))]
    mapping = dict(zip(set(episodes), num))
    comm = list(map(mapping.get, episodes))

    plot_mod_heatmap(
        corr, communities=np.asarray(comm), inds=range(len(corr)), edgecolor="white"
    )
    logger.info("Plot is generated")
    plt.savefig(
        f"/scratch/flesp/figures/" f"{kind}ISC_correlation_matrix_with_anat.png",
        bbox_inches="tight",
    )


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("figures_dir", type=click.Path(exists=True))
@click.option("--taskwise", type=bool)
@click.option("--kind", type=str)
@click.option("--slices", type=bool)
def plot_axial_slice(
    data_dir, figures_dir, tasks=tasks, taskwise=False, kind="temporal", slices=False
):
    """
    Plot axial slice.

    This
    Parameters
    ----------
    kind : str
        Kind of ISC, must be in ['spatial', 'temporal']
    data_dir : str
        The path to the postprocessed data directory on disk.
        Should contain all generated ISC maps.
    """
    logger = logging.getLogger(__name__)
    if kind not in ["spatial", "temporal"]:
        err_msg = "Unrecognized ISC type! Must be spatial or temporal"
        raise ValueError(err_msg)
    if taskwise is True:
        logger.info("Taskwise visualization")
        for task in tasks:
            files = glob.glob(f"{data_dir}/{task}/*.nii.gz")
            isc_volumes = []
            if slices is True:
                for fn in files:
                    isc_volumes.append(image.mean_img(fn))
            else:
                isc_volumes = [image.mean_img(files)]
            logger.info(f"Plotting {task}")
            # NOTE: threshold may need to be adjusted for each decoding task
            for idx, average in enumerate(isc_volumes):
                plotting.plot_stat_map(
                    average,
                    threshold=0.2,
                    vmax=0.75,
                    symmetric_cbar=False,
                    display_mode="z",
                    cut_coords=[-24, -6, 3, 25, 37, 51, 65],
                    title=f"{kind} ISC on {task}",
                )
                fn = str(f"{figures_dir}/{task}/{kind}ISC_on_{task}seg{idx:02d}.png")
                if os.path.exists(fn):
                    os.remove(fn)
                try:
                    plt.savefig(fn, bbox_inches="tight")
                except FileNotFoundError:
                    logger.info(f"Creating path for {task}")
                    os.mkdir(f"{figures_dir}/{task}/")
                    plt.savefig(fn, bbox_inches="tight")

            plt.close("all")
    else:
        logger.info("All tasks averaging")
        files = glob.glob(f"{data_dir}/*/*.nii.gz")
        average = image.mean_img(files, verbose=5, n_jobs=-1)

        # NOTE: threshold may need to be adjusted for each decoding task
        plotting.plot_stat_map(
            average,
            threshold=0.2,
            vmax=1,
            cmap="magma",
            draw_cross=True,
            annotate=False,
            symmetric_cbar=True,
            display_mode="z",
            cut_coords=[-24, -6, 3, 25, 37, 51, 65],
        )
        plt.savefig(f"{figures_dir}/{kind}ISC_on_All.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    mosaic_surface_isc_plots()
    surface_isc_plots()
    plot_corr_mtx()
    plot_axial_slice()
