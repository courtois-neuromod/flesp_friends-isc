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


from nilearn.maskers import MultiNiftiMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.datasets import fetch_atlas_harvard_oxford
import nibabel as nib
from brainiak import io


def multi_nifti_mask(scans, masks, confounds, fwhm=None, roi=False):
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
    # niftimask and clean data
    if roi is True:
        atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',
                                           symmetric_split=True)
        maskers = NiftiLabelsMasker(labels_img=atlas.maps, t_r=1.49,
                                    standardize=False, detrend=True,
                                    high_pass=0.01, low_pass=0.1,
                                    smoothing_fwhm=fwhm)
    else:
        maskers = MultiNiftiMasker(mask_img=masks, t_r=1.49,
                                   standardize=False, detrend=True,
                                   high_pass=0.01, low_pass=0.1,
                                   smoothing_fwhm=fwhm)

    cleaned = maskers.fit_transform(scans, confounds=confounds)
    return maskers.inverse_transform(cleaned)


def create_data_dictionary(data_dir, sessions=None, verbose=False):
    """
    Get all the participant file names

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
    if sessions is None:
        sessions = []
        for sub in subs:
            for ses in glob.glob(f'{data_dir}/{sub}/ses-*/'):
                sessions.append(ses[-8:-1])
                sessions.sort()
            data_dict[sub] = sessions
    else:
        for sub in subs:
            data_dict[sub] = sessions

    # Collect all file names
    nifti_fnames = []
    mask_fnames = []
    for sub_dirs in data_dict:
        for ses in data_dict[sub_dirs]:           
            nifti_fnames.extend(glob.glob(f'{data_dir}/{sub}/{ses}/func/*'
                                'MNI152NLin2009cAsym_desc-preproc_bold.nii.*'))
            mask_fnames.extend(glob.glob(f'{data_dir}/{sub}/{ses}/func/*'
                                         'MNI152NLin2009cAsym_desc-brain_mask'
                                         '.nii.gz'))
    if verbose:
        pprintpp.pprint(data_dict)
        pprintpp.pprint(nifti_fnames)

    return nifti_fnames, mask_fnames


def multisubject_process_episodes(nifti_fnames, output_filepath, episodes,
                                  mask_fnames, fwhm, roi):
    """
    Process episodes.

    Runs through the data dictionary to iterate through episodes. Each key is
    an episode and and it contains a list of path/filename for each subject's
    neuroimaging acquisitions.

    Parameters
    ----------
    nifti_fnames [list]:
        returned by create_data_dictionary
    output_filepath [path]:
        output path
    episodes [list]:
        list given by csv file
    mask_fnames [dictionary]:
        returned by create_data_dictionary
    fwhm [int]:
        smoothing parameter
    roi [bool]: False
        whether to process data ROI-wise
    """
    # NOTE : consider making group assigments for bootsraps
    # group_assignment_dict =
    # {task_name: i for i, task_name in enumerate(episodes)}
    if episodes is None:
        # write an error
        raise ValueError('Missing episodes list')
    fnames = {}
    masked_images = {}
    images = {}
    confs = {}

    for task_name in episodes:
        # list data as dict values for each sub and each item is episode
        fnames[task_name] = fnmatch.filter(nifti_fnames, f'*{task_name}*')
        # loads confounds files
        confs[task_name] = load_confounds_strategy(fnames[task_name],
                                                   denoise_strategy='simple',
                                                   global_signal='basic')
        images[task_name] = io.load_images(fnames[task_name])

        masked_images[task_name] = multi_nifti_mask(
                                scans=images[task_name],
                                masks=fnmatch.filter(mask_fnames,
                                                     f'*{task_name}*'),
                                confounds=confs,
                                smoothing_fwhm=fwhm,
                                roi=roi)

        # convert NaNs to zero0
        masked_images[task_name][np.isnan(masked_images[task_name])] = 0
        # print the shape
        print(f"Data : {task_name} \t"
              f'shape:{np.shape(masked_images[task_name])}')

        tmpl = 'space-MNI152NLin2009cAsym_desc-preproc_bold'
        if roi:
            tmpl = 'ROI_atlas_harvard_oxford' + tmpl
        postproc_fname = str(f'{task_name}/{masked_images[task_name]}'
                             f'_task-{task_name}_'
                             f'{tmpl}.hdf5').replace('desc-preproc',
                                                     f'desc-fwhm{fwhm}')

    del images
    nib.save(masked_images[task_name], os.path.join(output_filepath,
                                                    postproc_fname))
    return postproc_fname


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = os.path.join(project_dir, input_filepath)
    logger.info(f'Looking for data in :{data_dir}')
    nifti_names, mask_names = create_data_dictionary(
            data_dir, verbose=True)
    episodes = pd.read_csv(f'{project_dir}/episodes.csv',
                    delimiter=',')
    pprintpp.pprint(episodes)
    #multisubject_process_episodes(nifti_names, output_filepath, episodes,
     #                             mask_names, fwhm=6, roi=False)


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
