"""Viz."""
import glob
import os
import click
import logging
import itertools
from nilearn import image, plotting, surface, input_data
from nilearn.datasets import fetch_surf_fsaverage
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

fsaverage = fetch_surf_fsaverage()
mask_name = 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
brain_mask = nib.load(mask_name)
subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']
episodes = glob.glob(f"/scratch/flesp/data/iscs/*/")
tasks = []
for task in sorted(episodes):
    tasks.append(task[-13:-1])


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
def surface_isc_plots(data_dir, subject, tasks=tasks,
                      views=['lateral', 'medial'], hemi='left',
                      threshold=0.2, vmax=None):
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
    for view, task in itertools.product(views, tasks):
        logger.info(f"{view} | {task}")
        isc_files = sorted(glob.glob(f'{data_dir}/{task}/*.nii.gz'))
        average_isc = image.mean_img(isc_files)
        logger.info("Averaged BOLD images")
        # plot left hemisphere
        texture = surface.vol_to_surf(average_isc, fsaverage.pial_left)
        plotting.plot_surf_stat_map(
            fsaverage.pial_left, texture, hemi=hemi,
            colorbar=True, threshold=threshold, vmax=vmax,
            bg_map=fsaverage.sulc_left, view=view,
            title=f"{subject} {task}")
        try:
            plt.savefig(f'/scratch/flesp/figures/{task}/'
                        f'left_{view}_surfplot_ISC_on_{task}_{subject}.png',
                        bbox_inches='tight')
        except FileNotFoundError:
            logger.info(f"Creating path for {task}")
            os.mkdir(f'/scratch/flesp/figures/{task}/')
            plt.savefig(f'/scratch/flesp/figures/{task}/'
                        f'left_{view}_surfplot_ISC_on_{task}_{subject}.png',
                        bbox_inches='tight')
        # plot right hemisphere
        texture = surface.vol_to_surf(average_isc, fsaverage.pial_right)
        plotting.plot_surf_stat_map(
            fsaverage.pial_right, texture, hemi=hemi,
            colorbar=True, threshold=threshold, vmax=vmax,
            bg_map=fsaverage.sulc_right, view=view)
        plt.savefig(f'/scratch/flesp/figures/{task}/'
                    f'right_{view}_surfplot_ISC_on_{task}_{subject}.png',
                    bbox_inches='tight')
        plt.close


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
def plot_corr_mtx(data_dir, mask_img, kind='temporal'):
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
    from netneurotools.plotting import plot_mod_heatmap
    if kind not in ['spatial', 'temporal']:
        err_msg = 'Unrecognized ISC type! Must be spatial or temporal'
        raise ValueError(err_msg)

    isc_files = sorted(glob.glob(f'{data_dir}/*/*.nii.gz'))
    masker = input_data.NiftiMasker(mask_img=mask_img)

    isc = [masker.fit_transform(i).mean(axis=0) for i in isc_files]
    corr = np.corrcoef(np.row_stack(isc))

    # our 'communities' are which film was presented
    segment = [i.split('_task-')[-1].strip('.nii.gz') for i in isc_files]
    num = [i for i, m in enumerate(set(segment))]
    mapping = dict(zip(set(segment), num))
    comm = list(map(mapping.get, segment))

    plot_mod_heatmap(corr, communities=np.asarray(comm),
                     inds=range(len(corr)), edgecolor='white')
    plt.savefig(f'{kind}ISC_correlation_matrix_with_anat.png',
                bbox_inches='tight')


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
def plot_axial_slice(tasks, data_dir, kind='temporal'):
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
    if kind not in ['spatial', 'temporal']:
        err_msg = 'Unrecognized ISC type! Must be spatial or temporal'
        raise ValueError(err_msg)

    for task in tasks:
        files = glob.glob(f'data_dir/{task}/*.nii.gz')
        average = image.mean_img(files)

        # NOTE: threshold may need to be adjusted for each decoding task
        plotting.plot_stat_map(
            average,
            threshold=0.1, vmax=0.75, symmetric_cbar=False,
            display_mode='z', cut_coords=[-24, -6, 7, 25, 37, 51, 65]
        )
        plt.savefig(f'/scratch/flesp/figures/{task}/{kind}ISC_on_{task}.png',
                    bbox_inches='tight')


if __name__ == '__main__':
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    surface_isc_plots()
    plot_corr_mtx()
    plot_axial_slice(tasks)
