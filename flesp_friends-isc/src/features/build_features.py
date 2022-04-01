import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import glob
import numpy as np
from brainiak.isc import isc
from brainiak import image, io
import nibabel as nib

subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']

mask_name = 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
brain_mask = io.load_boolean_mask(mask_name)
coords = np.where(brain_mask)
brain_nii = nib.load(mask_name)
print("Loaded mask")


@click.command()
@click.argument('postproc_path', type=click.Path(exists=True))
@click.argument('isc_map_path', type=click.Path(exists=True))
def map_isc(postproc_path, isc_map_path, pairwise=False):
    """
    Compute ISC for brain data.

    note
    """
    # specify data path (leads to subdi
    logger = logging.getLogger(__name__)
    logger.info('Starting Temporal ISC workflow')
    tasks = glob.glob(f"{postproc_path}/*/")
    # walks subdirs with taks name (task-s01-e01a)
    for task in sorted(tasks):
        task = task[-13:-1]
        logger.info(f'Importing data')
        files = sorted(glob.glob(f'{postproc_path}/{task}/*.nii.gz*'))

        images = io.load_images(files)
        logger.info("Loaded files")
        masked_imgs = image.mask_images(images, brain_mask)
        logger.info("Masked images")

        # compute ISC
        bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                masked_imgs, len(files))
        logger.info(f"Correctly imported masked images for {len(files)} subjs")

        # replace nans
        bold_imgs[np.isnan(bold_imgs)] = 0
        # compute ISC
        logger.info("Computing ISC")
        isc_imgs = isc(bold_imgs, pairwise=pairwise)

        # save ISC maps per subject
        for n, subj in enumerate(subjects):
            # Make the ISC output a volume
            isc_vol = np.zeros(brain_nii.shape)
            # Map the ISC data for the first participant into brain space
            isc_vol[coords] = isc_imgs[n, :]
            # make a nii image of the isc map
            isc_nifti = nib.Nifti1Image(
                isc_vol, brain_nii.affine, brain_nii.header
            )

            # Save the ISC data as a volume
            logger.info("Saving images")
            try:
                nib.save(isc_nifti, f'{isc_map_path}/{task}/temporalISC_{task}'
                                    f'_{subj}.nii.gz')
            except FileNotFoundError:
                os.path.mkdir(f"{isc_map_path}/{task}")
                nib.save(isc_nifti, f'{isc_map_path}/{task}/temporalISC_{task}'
                                    f'_{subj}.nii.gz')
        # free up memory
        del bold_imgs, isc_imgs
        logger.info(f"Done workflow for {task}")

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
