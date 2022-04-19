"""ISC workflow."""
import os
import click
import fnmatch
import logging
from dotenv import find_dotenv, load_dotenv
import glob
import numpy as np
from brainiak.isc import isc, isfc
from brainiak import image, io
import nibabel as nib
#from nilearn.datasets import fetch_atlas_harvard_oxford

subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']


@click.command()
@click.argument('postproc_path', type=click.Path(exists=True))
@click.argument('isc_map_path', type=click.Path(exists=True))
@click.option('--roi', type=bool)
@click.option('--kind', type=str)
@click.option('--pairwise', type=bool)
@click.option('--drop', type=str)
def map_isc(postproc_path, isc_map_path, kind='temporal',
            pairwise=None, roi=False, drop=None):
    """
    Compute ISC for brain data.

    note
    """
    # specify data path (leads to subdi
    logger = logging.getLogger(__name__)
    logger.info(f'Starting {kind} ISC workflow')
    tasks = glob.glob(f"{postproc_path}/*/")
    # walks subdirs with taks name (task-s01-e01a)
    for task in sorted(tasks):
        task = task[-13:-1]
        logger.info(f'Importing data')
        files = sorted(glob.glob(f'{postproc_path}/{task}/*.nii.gz*'))

        if drop is None:
            logger.info('Considering all subjects for ISCs')
        else:
            fn_to_remove = fnmatch.filter(files, f"*{drop}*")
            logger.info(f'Not considering all subjects for ISCs \n'
                        f'Removing : {fn_to_remove}')
            files.remove(fn_to_remove)
        for fn in files:
            _, fn = os.path.split(fn)
            logger.info(fn[:6])
        # Fetch images and mask them
        images = io.load_images(files)
        logger.info("Loaded files")
        # Parcel space or not
        if roi is True:
            logger.info(f"Masking data using labels")
            # atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',
                                               # data_dir="/scratch/flesp/",
                                               # symmetric_split=True)
            mask_name = "scratch/flesp/fsl/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
            brain_nii = nib.load(mask_name)
            brain_mask = io.load_boolean_mask(mask_name)
            masked_imgs = image.mask_images(images, brain_mask)
            # figure out missing rois
            #row_has_nan = np.zeros(shape=(len(atlas.labels)-1,), dtype=bool)
            # Check for nans in each images and listify
            #for n in range(len(files)):
            #    row_has_nan_ = np.any(np.isnan(masked_imgs[:, :, n]), axis=0)
            #    row_has_nan[row_has_nan_] = True
            # coordinates/regions that contain nans
            #coords = np.logical_not(row_has_nan)
            #rois_filtered = np.array(atlas.labels[1:])[coords]
            #logger.info(f"{rois_filtered}")
            #masked_imgs = masked_imgs[:, coords, :]
        # here we render in voxel space
        else:
            mask_name = 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
            brain_mask = io.load_boolean_mask(mask_name)
            brain_nii = nib.load(mask_name)
            coords = np.where(brain_mask)
            masked_imgs = image.mask_images(images, brain_mask)
            logger.info("Masked images")

        # compute ISC
        try:
            bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                masked_imgs, len(files))
        except ValueError:
            logger.info(f"Can't perform MaskedMultiSubjectData on {task}")
            continue
        logger.info(f"Correctly imported masked images for {len(files)} subjs"
                    "\n------------------------------------------------------")
        # replace nans
        bold_imgs[np.isnan(bold_imgs)] = 0

        # Computing ISC
        logger.info("\n"
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                    f"Computing {kind} ISC on {task}\n"
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if pairwise:
            logger.info(f"{kind} ISC with pairwise approach")
        else:
            logger.info(f"{kind} ISC with Leave-One-Out approach")
        #
        if kind == 'temporal':
            isc_imgs = isc(bold_imgs, pairwise=pairwise)
        elif kind == 'spatial':
            isc_imgs = isfc(bold_imgs, pairwise=pairwise)
        else:
            logger.info(f"Cannot compute {kind} ISC on {task}")
            continue
        logger.info("Saving images")
        if pairwise is None:
            # save ISC maps per subject
            for n, fn in enumerate(files):
                _, sub = os.path.split(fn)
                logger.info(sub[:6])
                # Make the ISC output a volume
                isc_vol = np.zeros(brain_nii.shape)
                # Map the ISC data for the first participant into brain space
                isc_vol[coords] = isc_imgs[n, :]
                # make a nii image of the isc map
                isc_nifti = nib.Nifti1Image(
                    isc_vol, brain_nii.affine, brain_nii.header
                )
                # Save the ISC data as a volume
                if roi is True:
                    try:
                        nib.save(isc_nifti, f'{isc_map_path}/{task}/{sub}'
                                            f'_{task}_ROI{kind}ISC.nii.gz')
                    except FileNotFoundError:
                        os.mkdir(f"{isc_map_path}/{task}")
                        nib.save(isc_nifti, f'{isc_map_path}/{task}/{sub}'
                                            f'_{task}_ROI{kind}ISC.nii.gz')
                else:
                    try:
                        nib.save(isc_nifti, f'{isc_map_path}/{task}/{sub}'
                                            f'_{task}_{kind}ISC.nii.gz')
                    except FileNotFoundError:
                        os.mkdir(f"{isc_map_path}/{task}")
                        nib.save(isc_nifti, f'{isc_map_path}/{task}/{sub}'
                                            f'_{task}_{kind}ISC.nii.gz')
        else:
            c = 0
            for n, fn in enumerate(files):
                _, sub_a = os.path.split(fn)
                for m in range(n+1, len(files)):
                    _, sub_b = os.path.split(fn[m])
                    logger.info(f"{sub_a[:6]} | {sub_b[:6]}")
                    pair = f"{sub_a[:6]}"+f"-{sub_b[:6]}"
                    # Make the ISC output a volume
                    isc_vol = np.zeros(brain_nii.shape)
                    # Map the ISC data for the first participant into brain space
                    isc_vol[coords] = isc_imgs[c, :]
                    # make a nii image of the isc map
                    isc_nifti = nib.Nifti1Image(
                        isc_vol, brain_nii.affine, brain_nii.header
                    )
                    try:
                        nib.save(isc_nifti, f'{isc_map_path}/{task}/{pair}'
                                            f'_{task}_{kind}ISC.nii.gz')
                    except FileNotFoundError:
                        os.mkdir(f"{isc_map_path}/{task}")
                        nib.save(isc_nifti, f'{isc_map_path}/{task}/{pair}'
                                            f'_{task}_{kind}ISC.nii.gz')
                    c += 1

        # free up memory
        del bold_imgs, isc_imgs
        logger.info("\n"
                    "------------------------------------------------------\n"
                    f"          Done workflow for {task}             "
                    "\n------------------------------------------------------")


if __name__ == '__main__':
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    map_isc()
