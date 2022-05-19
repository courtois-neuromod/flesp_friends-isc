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
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import concat_imgs, index_img

subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']


@click.command()
@click.argument('postproc_path', type=click.Path(exists=True))
@click.argument('isc_map_path', type=click.Path(exists=True))
@click.option('--roi', type=bool)
@click.option('--kind', type=str)
@click.option('--pairwise', type=bool)
@click.option('--drop', type=str)
@click.option('--slices', type=bool)
def map_isc(postproc_path, isc_map_path, kind='temporal',
            pairwise=False, roi=False, drop=None, slices=False):
    """
    Compute ISC for brain data.

    note
    """
    # specify data path (leads to subdi
    logger = logging.getLogger(__name__)
    logger.info(f'Starting {kind} ISC workflow')
    tasks = glob.glob(f"{postproc_path}/*/")

    # mask and info
    mask_name = 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
    brain_mask = io.load_boolean_mask(mask_name)
    brain_nii = nib.load(mask_name)
    coords = np.where(brain_mask)

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
            files.remove(fn_to_remove[0])
        for fn in files:
            _, fn = os.path.split(fn)
            logger.info(fn[:6])

        # Parcel space or not
        if roi is True:
            logger.info(f"Masking data using labels")
            mask_name = "/scratch/flesp/fsl/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
            brain_nii = nib.load(mask_name)
            brain_mask = brain_nii.get_fdata()
            masker = NiftiLabelsMasker(labels_img=brain_nii.get_filename(),
                                       standardize=True, verbose=5)
            bold_imgs = []
            for fn in files:
                timeserie = masker.fit_transform(fn)
                bold_imgs.append(timeserie)

        # here we render in voxel space
        # Option to segment run in smaller windows
        elif slices is True:
            masked_imgs = []
            sub_sliced = {}
            lng = 100
            logger.info(f"Segmenting in slices of length {lng} TRs")
            # Fetch images
            for i, fn in enumerate(files):
                img = nib.load(fn)
                timeserie = img.get_fdata()
                imgs_sub = []
                # slice them subject-wise
                for idx in range(0, timeserie.shape[3]-lng, 50):
                    slx = slice(0 + idx, lng + idx)
                    sliced = nib.Nifti1Image(timeserie[:, :, :, slx],
                                             brain_nii.affine)
                    imgs_sub.append(sliced)
                sub_sliced[i] = imgs_sub
            # start by first segment in each subject and iterate
            for segment in range(len(sub_sliced[0])):
                ls_imgs = []
                # assemble a temporary list for each segment containing all sub
                for sub in sub_sliced:
                    ls_imgs.append(sub_sliced[sub][segment])
                # Mask every subject's segment and append in list
                masked_imgs.append(image.mask_images(ls_imgs, brain_mask))

        # mask the whole run
        else:
            images = io.load_images(files)
            masked_imgs = image.mask_images(images, brain_mask)
            logger.info("Masked images")

            try:
                bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                    masked_imgs, len(files))
                # replace nans
                bold_imgs[np.isnan(bold_imgs)] = 0
                logger.info(
                    f"Correctly imported masked images for {len(files)} subjs"
                    "\n------------------------------------------------------")
            except ValueError:
                logger.info(f"Can't perform MaskedMultiSubjectData on {task}")
                continue
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
            if slices is False:
                isc_imgs = [isc(bold_imgs, pairwise=pairwise)]
            else:
                isc_imgs = []
                for obj in masked_imgs:
                    bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                        obj, len(files))
                    isc_seg = isc(bold_imgs, pairwise=pairwise)
                    isc_imgs.append(isc_seg)
        elif kind == 'spatial':
            isc_imgs = isfc(bold_imgs, pairwise=pairwise)
        else:
            logger.info(f"Cannot compute {kind} ISC on {task}")
            continue
        logger.info("Saving images")
        if pairwise is False:
            # save ISC maps per subject
            for n, fn in enumerate(files):
                _, sub = os.path.split(fn)
                logger.info(sub[:6])
                # Make the ISC output a volume
                isc_vol = np.zeros(brain_nii.shape)
                # iterate through segments
                for idx, isc_seg in enumerate(isc_imgs):
                    logger.info(idx)
                    # Map the ISC data for each participant into 3d space
                    isc_vol[coords] = isc_seg[n, :]
                    # make a nii image of the isc map
                    isc_nifti = nib.Nifti1Image(
                        isc_vol, brain_nii.affine, brain_nii.header
                    )
                    # Save the ISC data as a volume
                    if roi is True:
                        try:
                            nib.save(isc_nifti, f'{isc_map_path}/{task}/'
                                                f'{sub[:6]}_{task}seg{idx:02d}'
                                                f'_ROI{kind}ISC.nii.gz')
                        except FileNotFoundError:
                            os.mkdir(f"{isc_map_path}/{task}")
                            nib.save(isc_nifti, f'{isc_map_path}/{task}/'
                                                f'{sub[:6]}_{task}seg{idx:02d}'
                                                f'_ROI{kind}ISC.nii.gz')
                    else:
                        try:
                            nib.save(isc_nifti, f'{isc_map_path}/{task}/'
                                                f'{sub[:6]}_{task}seg{idx:02d}'
                                                f'_{kind}ISC.nii.gz')
                        except FileNotFoundError:
                            os.mkdir(f"{isc_map_path}/{task}")
                            nib.save(isc_nifti, f'{isc_map_path}/{task}/'
                                                f'{sub[:6]}_{task}seg{idx:02d}'
                                                f'_{kind}ISC.nii.gz')
            # free up memory
            del masked_imgs, isc_imgs
        else:
            c = 0
            for n, fn in enumerate(files):
                _, sub_a = os.path.split(fn)
                for m in range(n+1, len(files)):
                    _, sub_b = os.path.split(files[m])
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
