# pylint: disable=logging-format-interpolation
"""Brain-ISC workflow."""
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
import pandas as pd

subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
TR=1.49
events_files_path = "~/projects/rrg-pbellec/flesp_captions-embeddings/friends_annotations/annotation_results/manual_segmentation/"

# mask and info
mask_name = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
brain_mask = io.load_boolean_mask(mask_name)
brain_nii = nib.load(mask_name)
coords = np.where(brain_mask)

def load_events(events_file):
    """
    Loads event onsets and durations from a TSV file.

    Parameters
    ----------
    events_file : str
        Path to the TSV file containing the event information.

    Returns
    -------
    events : ndarray, shape (n_events, 2)
        Array containing the event onsets and durations.
    """
    events_df = pd.read_csv(events_file, sep='\t')
    onsets = events_df['onset'].values
    durations = events_df['duration'].values
    events = np.column_stack((onsets, durations))
    return events


def _save_pair_feature_img(isc_imgs, isc_map_path, task, kind, files, roi):
    """
    Pairwise save function isc volumes either as numpy arrays or nifti file
    """
    logger = logging.getLogger(__name__)
    # save ISC maps per pairs of subject
    for idx_seg, isc_seg in enumerate(isc_imgs):
        counter = 0
        # define pairs based on filenames
        for n, fn in enumerate(files):
            _, sub_a = os.path.split(fn)
            for m in range(n + 1, len(files)):
                _, sub_b = os.path.split(files[m])
                # log pair and segment
                logger.info(f"Segment {idx_seg:02d} | {sub_a[:6]} | {sub_b[:6]}")
                pair = f"{sub_a[:6]}" + f"-{sub_b[:6]}"
                # save numpy array
                if roi is True:
                    fn = f"{pair}_{task}seg{idx_seg:02d}ROI{kind}ISC.npy"
                    try:
                        np.save(f"{isc_map_path}/{task}/{fn}", isc_seg[counter, :])
                    # create folder for episode if not exist
                    except FileNotFoundError:
                        os.mkdir(f"{isc_map_path}/{task}/")
                        np.save(f"{isc_map_path}/{task}/{fn}", isc_seg[counter, :])
                else:
                    # Make the ISC output a volume
                    isc_vol = np.zeros(brain_nii.shape)
                    # Map the ISC data for the first participant into brain space
                    isc_vol[coords] = isc_seg[counter, :]
                    # make a nii image of the isc map
                    isc_nifti = nib.Nifti1Image(
                        isc_vol, brain_nii.affine, brain_nii.header
                    )
                    if not os.path.exists(f"{isc_map_path}/{task}"):
                        os.mkdir(f"{isc_map_path}/{task}")

                    nib.save(
                        isc_nifti,
                        f"{isc_map_path}/{task}/{pair}_{task}seg{idx_seg:02d}_{kind}ISC.nii.gz",
                    )
                counter += 1


def _save_sub_feature_img(isc_imgs, isc_map_path, task, kind, files, roi):
    """
    Subject-wise save function isc volumes either as numpy arrays of nifti files
    """
    logger = logging.getLogger(__name__)
    # save ISC maps per subject
    for n, fn in enumerate(files):
        _, sub = os.path.split(fn)
        logger.info(sub[:6])
        # Make the ISC output a volume
        isc_vol = np.zeros(brain_nii.shape)
        # iterate through segments
        for idx, isc_seg in enumerate(isc_imgs):
            # save numpy array
            if roi is True:
                fn = f"{sub[:6]}_{task}seg{idx:02d}ROI{kind}ISC.npy"
                try:
                    np.save(f"{isc_map_path}/{task}/{fn}", isc_seg[n, :])
                # create folder for episode if not exist
                except FileNotFoundError:
                    os.mkdir(f"{isc_map_path}/{task}/")
                    np.save(f"{isc_map_path}/{task}/{fn}", isc_seg[n, :])
                continue
            else:
                # Map the ISC data for each participant into 3d space
                isc_vol[coords] = isc_seg[n, :]
                # make a nii image of the isc map
                isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)
                # Save the ISC data as a volume
                if not os.path.exists(f"{isc_map_path}/{task}"):
                    os.mkdir(f"{isc_map_path}/{task}")

                fn = f"{sub[:6]}_{task}seg{idx:02d}_{kind}ISC.nii.gz"

                nib.save(isc_nifti, f"{isc_map_path}/{task}/{fn}")

def __slice_timeseries_subjectwise(timeserie, affine, range_step, events=None):
    """
    Slices a 4D timeseries along the 4th dimension according to a range step.
    If events are provided, slices the timeseries according to the event onsets and durations.
    """
    # slice them based on arbiratry range step
    if events is None:
        for idx in range_step:
            slx = slice(0 + idx, timeserie.shape[3] + idx)
            sliced = nib.Nifti1Image(timeserie[:, :, :, slx], affine)
            imgs_sub.append(sliced)
    # slice them based on manual scene segmentation events
    else:
        onsets = events['onset'].values
        durations = events['duration'].values
        valid_idx = np.where(durations>=45)
        for onset, duration in zip(onsets[valid_idx], durations[valid_idx]):
            onset_idx = int(round(onset / TR))
            duration_idx = int(round(duration / TR))
            sliced = timeserie[:, :, :, onset_idx:onset_idx+duration_idx-1]
            sliced = nib.Nifti1Image(sliced, affine)
            imgs_sub.append(sliced)
    # associate to key in dict for 1 sub
    return imgs_sub

def _slice_img_timeseries(files, lng, affine=brain_nii.affine, roi=False, events=None):
    """
    Slice 4D timeseries.

    Parameters
    ----------
    files : list
        List of files to be sliced.
    lng : int
        Length of the slice.
    affine : array
        Affine matrix of the image.
    roi : bool
        If True, the input is a numpy array.
    events : bool
        If True, the input is a numpy array.
    Returns
    -------
    masked_imgs : list
        List of sliced images.
    """
    masked_imgs = []
    sub_sliced = {}
    # define slices for events
    if event is True:
        # load events file
        task_filename_pattern = files[0].split("_")[1]
        task_filename_pattern = task_filename_pattern.split("-s0")[-1]
        event_file = fnmatch.filter(glob.glob(f"{events_files_path}*/*manualseg.tsv"),f"*{task_filename_pattern}*")[0]
        events = pd.read_csv(event_file, sep='\t')
    else:
        events = None

    # Fetch images
    for i, processed in enumerate(files):
        # load image
        if roi is False:
            img = nib.load(processed)
            timeserie = img.get_fdata()
            timeserie_len = timeserie.shape[3]
        # load numpy array
        else:
            timeserie = processed
            timeserie_len = int(len(timeserie))
        # define range of slices, make overlaps with step size of half the length for
        # this specific segment length
        if lng == 100:
            range_step = range(0, int(timeserie_len - lng), int(lng / 2))
        elif lng == 30:
            range_step = range(0, int(timeserie_len - lng), lng)
        else:
            range_step = None
        # slice them subject-wise
        sub_sliced[i] = __slice_timeseries_subjectwise(timeserie, affine, range_step, events)
    # start by first segment in each subject and iterate
    for segment in range(len(sub_sliced[0])):
        ls_imgs = []
        # assemble a temporary list for each segment containing all sub
        for sub in sub_sliced:
            ls_imgs.append(sub_sliced[sub][segment])
        if roi is False:
            # Mask every subject's segment and append in list
            masked_imgs.append(image.mask_images(ls_imgs, brain_mask))
        else:
            masked_imgs.append(ls_imgs)

    del sub_sliced

    return masked_imgs


@click.command()
@click.argument("postproc_path", type=click.Path(exists=True))
@click.argument("isc_map_path", type=click.Path(exists=True))
@click.option("--roi", type=bool)
@click.option("--kind", type=str)
@click.option("--pairwise", type=bool)
@click.option("--drop", type=str)
@click.option("--slices", type=bool)
@click.option("--lng", type=int)
@click.option("--events", type=bool)
def map_isc(
    postproc_path,
    isc_map_path,
    kind="temporal",
    pairwise=False,
    roi=False,
    drop=None,
    slices=False,
    lng=30,
):
    """
    Load timeseries, slice them, compute ISCs and save timeseries.

    Arguments:
    ------------
    postproc_path: str
        Path to preprocessed data
    isc_map_path: str
        Path to save ISC maps
    kind: str
        Kind of ISC to compute (temporal, spatial, or both)
    pairwise: bool
        Compute pairwise ISC
    roi: bool
        Compute ISC on ROIs
    drop: str
        Drop subjects from ISC computation
    slices: bool
        Slice timeseries
    lng: int
        Length of timeseries to slice
    
    Saves ISC maps in nifti or numpy format
    """    
    # specify data path (leads to subdi
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {kind} ISC workflow")
    tasks = glob.glob(f"{postproc_path}/*/")

    # walks subdirs with taks name (task-s01-e01a)
    for task in sorted(tasks):
        task = task[-13:-1]
        logger.info("Importing data")
        # fetch files
        if roi is True:
            files = sorted(glob.glob(f"{postproc_path}/{task}/*256.npy"))
        else:
            files = sorted(glob.glob(f"{postproc_path}/{task}/*.nii.gz*"))
        # log subjects
        if drop is None:
            logger.info("Considering all subjects for ISCs")
        else:
            fn_to_remove = fnmatch.filter(files, f"*{drop}*")
            logger.info(
                f"Not considering all subjects for ISCs \n" f"Removing : {fn_to_remove}"
            )
            files.remove(fn_to_remove[0])
        for fn in files:
            _, fn = os.path.split(fn)
            logger.info(fn[:6])

        # do not compute if alone
        if len(files) == 1:
            logger.info(
                f"{task} is left out because only {len(files)} files accessible"
            )
            continue
        # Parcel space or not
        if roi is True:
            logger.info("Loading ROIs data")
            bold_imgs = []
            for fn in files:
                bold_imgs.append(np.load(fn))
            if slices is True:
                masked_imgs = _slice_img_timeseries(bold_imgs, lng, roi=roi)

        # here we render in voxel space
        # Option to segment run in smaller windows
        elif slices is True and roi is False:
            logger.info(f"Segmenting in slices of length {lng} TRs")
            masked_imgs = _slice_img_timeseries(files, lng, events=events)

        # mask the whole run
        elif roi is False and slices is False:
            images = io.load_images(files)
            masked_imgs = image.mask_images(images, brain_mask)
            logger.info("Masked images")

            try:
                bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                    masked_imgs, len(files)
                )
                # replace nans
                bold_imgs[np.isnan(bold_imgs)] = 0
                logger.info(
                    f"Correctly imported masked images for {len(files)} subjs"
                    "\n------------------------------------------------------"
                )
            except ValueError:
                logger.info(f"Can't perform MaskedMultiSubjectData on {task}")
                continue
        # Computing ISC
        logger.info(
            "\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            f"Computing {kind} ISC on {task}\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        # log the method
        if pairwise:
            logger.info(f"{kind} ISC with pairwise approach")
        else:
            logger.info(f"{kind} ISC with Leave-One-Out approach")
        # Temporal ISC workflow either for sliced or not timeseries
        if kind == "temporal":
            # build a list even though it's a single ISC for that run
            if not slices:
                isc_imgs = [isc(bold_imgs, pairwise=pairwise)]
            # workflow for sliced timeseries
            else:
                isc_imgs = []
                # if we are in voxel space 
                if roi is False:
                    for niimg_obj in masked_imgs:
                        bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                            niimg_obj, len(files)
                        )
                        isc_seg = isc(bold_imgs, pairwise=pairwise)
                        isc_imgs.append(isc_seg)
                # if we are in parcel space
                else:
                    for bold_ts in masked_imgs:
                        isc_seg = isc(bold_ts, pairwise=pairwise)
                        isc_imgs.append(isc_seg)
        # Spatial ISC workflow
        elif kind == "spatial":
            isc_imgs = isfc(bold_imgs, pairwise=pairwise)
        # Cannot compute ISC of this kind yet
        else:
            logger.info(f"Cannot compute {kind} ISC on {task}\n choose temporal or spatial")
            continue

        logger.info("Saving images")
        # save data
        if pairwise is False:
            _save_sub_feature_img(isc_imgs, isc_map_path, task, kind, files, roi)
            # free up memory
            del bold_imgs, isc_imgs

        # if it's pairwise
        else:
            _save_pair_feature_img(isc_imgs, isc_map_path, task, kind, files, roi)
            # free up memory
            del bold_imgs, isc_imgs
        logger.info(
            "\n"
            "------------------------------------------------------\n"
            f"          Done workflow for {task}             "
            "\n------------------------------------------------------"
        )


if __name__ == "__main__":
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    map_isc()
