# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
import fnmatch
import pprintpp
import numpy as np
import pandas as pd
from nilearn.maskers import MultiNiftiMasker, NiftiMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.datasets import fetch_atlas_harvard_oxford
import nibabel as nib


def nifti_mask(scans, masks, confounds, fwhm, roi=False):
    """
    Mask the images.

    Cleans data, detrending,
    and high-pass filtering at 0.01Hz. Corrects for supplied
    confounds. Optionally smooths time series.

    Parameters
    ----------
    scan: niimg_like
        An in-memory niimg
    mask: str
        The (brain) mask within which to process data.
    confounds: np.ndarray
        Any confounds to correct for in the cleaned data set.
    """
    # ROI workflow
    if roi is True:
        atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',
                                           symmetric_split=True)
        maskers = NiftiLabelsMasker(labels_img=atlas.maps, t_r=1.49,
                                    standardize=False, detrend=True,
                                    high_pass=0.01, low_pass=0.1,
                                    smoothing_fwhm=fwhm)

        cleaned = maskers.fit_transform(scans, confounds=confounds)
        masked_imgs = maskers.inverse_transform(cleaned)
    # generic mask workflow
    elif isinstance(masks, list) is False:
        maskers = MultiNiftiMasker(mask_img=masks, t_r=1.49,
                                   standardize=False, detrend=True,
                                   high_pass=0.01, low_pass=0.1,
                                   smoothing_fwhm=fwhm)
        masked_imgs = maskers.inverse_transform(cleaned)
    # individual anatomical mask subject-wise
    else:
        masked_imgs = []
        for mask, bold, conf in zip(masks, scans, confounds):
            masker = NiftiMasker(mask_img=mask, t_r=1.49,
                                 standardize=False, detrend=True,
                                 high_pass=0.01, low_pass=0.1,
                                 smoothing_fwhm=fwhm)
            cleaned = masker.fit_transform(bold, confounds=conf[0])
            print(f"Fitted {os.path.basename(bold)}")
            masked_imgs.append(masker.inverse_transform(cleaned))

    return masked_imgs


def create_data_dictionary(data_dir, sessions=None, verbose=False):
    """
    Get all the participant file names.

    Parameters
    ----------
    data_dir_ [str]: the data root dir

    Return
    ----------
    nifti_fnames, mask_fnames [list]: file names for all subjs
    """
    data_dict = {}
    subs = []
    for sub in glob.glob(f'{data_dir}/sub-*/'):
        subs.append(sub[-7:-1])
        subs.sort()
    for sub in subs:
        sessions = []
        for ses in glob.glob(f'{data_dir}/{sub}/ses-*/'):
            sessions.append(ses[-8:-1])
            sessions.sort()
        data_dict[sub] = sessions

    # Collect all file names
    nifti_fnames = []
    mask_fnames = []
    for sub_dirs in data_dict:
        for ses in data_dict[sub_dirs]:
            nifti_fnames.extend(glob.glob(f'{data_dir}/{sub_dirs}/{ses}/func/*'
                                          'MNI152NLin2009cAsym_desc-preproc_bold'
                                          '.nii.gz'))

            mask_fnames.extend(glob.glob(f'{data_dir}/{sub_dirs}/{ses}/func/*'
                                         'MNI152NLin2009cAsym_desc-brain_mask'
                                         '.nii.gz'))
    if verbose:
        pprintpp.pprint(data_dict)

    return nifti_fnames, mask_fnames


def process_episodewise(fnames, output_filepath, task_name,
                        masks, fwhm, roi):
    """
    Process episodes.

    Runs through the data dictionary to iterate through episodes. Each key is
    an episode and and it contains a list of path/filename for each subject's
    neuroimaging acquisitions.

    Parameters
    ----------
    fnames [list]:
        returned by create_data_dictionary
    output_filepath [path]:
        output path
    fwhm [int]:
        smoothing parameter
    roi [bool]: False
        whether to process data ROI-wise
    """
    # NOTE : consider making group assigments for bootsraps
    # group_assignment_dict =
    # {task_name: i for i, task_name in enumerate(episodes)}
    # loads confounds files
    confs = []
    for nii in fnames:
        conf = load_confounds_strategy(nii, denoise_strategy='simple',
                                       motion='basic')
        confs.append(conf)

    masked_images = nifti_mask(scans=fnames,
                               masks=masks,
                               confounds=confs,
                               fwhm=fwhm,
                               roi=roi)

    # print the shape
    for i, img in enumerate(masked_images):
        sub = os.path.basename(fnames[i])[4:6]
        print(f"Task : {task_name} \n"
              f"Subject ID: {sub} \n"
              f'shape:{np.shape(img)}')

    tmpl = f'space-MNI152NLin2009cAsym_desc-fwhm{fwhm}'
    if roi:
        tmpl = 'ROI_atlas_harvard_oxford' + tmpl

    del confs
    for i, img in enumerate(masked_images):
        sub = os.path.basename(fnames[i])[:6]
        postproc_fname = str(f'{task_name}/{sub}'
                             f'_{task_name}_{tmpl}.nii.gz')
        fn = os.path.join(f"{output_filepath}", postproc_fname)
        try:
            nib.save(img, fn)
        except FileNotFoundError:
            os.mkdir(f"{output_filepath}/{task_name}")
            nib.save(img, fn)
        print(f"Saved {sub}, {task_name} under: {fn}")
    return postproc_fname


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Prepare dataset.

    I/O function to process all episodes.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = os.path.join(project_dir, input_filepath)
    logger.info(f'Looking for data in :{data_dir}')
    nifti_names, mask_names = create_data_dictionary(data_dir, verbose=True)
    episodes = list(pd.read_csv(f'{project_dir}/episodes.csv',
                                delimiter=',', header=None).iloc[:, 0])
    logger.info(f"Iterating through episodes : {episodes[:5]}...")
    # iterate through episodes
    for task_name in episodes:
        logger.info(f'Processing : {task_name}')
        # list data as dict values for each sub and each item is episode
        fnames = fnmatch.filter(nifti_names, f'*{task_name}*')
        masks = fnmatch.filter(mask_names, f'*{task_name}*')
        fnames.sort()
        masks.sort()
        process_episodewise(fnames, output_filepath, task_name,
                            masks, fwhm=6, roi=False)
        logger.info(f'Done processing : {task_name} \n'
                    '---------------------------------')


if __name__ == '__main__':
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    #subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
    main()
