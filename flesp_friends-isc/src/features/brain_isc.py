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
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=1)

# mask and info
#mask_name = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
mask_name = atlas.maps
brain_mask = io.load_boolean_mask(mask_name)
brain_nii = nib.load(mask_name)
coords = np.where(brain_mask)


def _save_pair_feature_img(isc_imgs, isc_map_path, task, kind, files, roi):
    """internal function that save img stat map of isc.

    Args:
        isc_imgs (list of img_obj): _description_
        isc_map_path (Path): Path to base dir
        task (str): name of task
        kind (str): type of ISC
        files (list of str): list of file paths
        roi (bool): whether or no
    """

    logger = logging.getLogger(__name__)
    # save ISC maps per pairs of subject
    for idx_seg, isc_seg in enumerate(isc_imgs):
        counter = 0
        for n_pair, file_fn in enumerate(files):
            _, sub_a = os.path.split(file_fn)
            for m_pair in range(n_pair + 1, len(files)):
                _, sub_b = os.path.split(files[m_pair])
                logger.info(f"Segment {idx_seg:02d} | {sub_a[:6]} | {sub_b[:6]}")
                pair = f"{sub_a[:6]}" + f"-{sub_b[:6]}"
                if roi is True:
                    fn_npy = f"{pair}_{task}seg{idx_seg:02d}ROI{kind}ISC.npy"
                    try:
                        np.save(f"{isc_map_path}/{task}/{fn_npy}", isc_seg[counter, :])
                    except FileNotFoundError:
                        os.mkdir(f"{isc_map_path}/{task}/")
                        np.save(f"{isc_map_path}/{task}/{fn_npy}", isc_seg[counter, :])
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
    """internal function to save ISC stat maps computed LOO.

    Args:
        isc_imgs (list): _description_
        isc_map_path (path): _description_
        task (st): _description_
        kind (str): _description_
        files (list): _description_
        roi (bool): _description_
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
            if roi is True:
                fn_npy = f"{sub[:6]}_{task}seg{idx:02d}ROI{kind}ISC.npy"
                try:
                    np.save(f"{isc_map_path}/{task}/{fn_npy}", isc_seg[n, :])
                except FileNotFoundError:
                    os.mkdir(f"{isc_map_path}/{task}/")
                    np.save(f"{isc_map_path}/{task}/{fn_npy}", isc_seg[n, :])
            else:
                # Map the ISC data for each participant into 3d space
                isc_vol[coords] = isc_seg[n, :]
                # make a nii image of the isc map
                isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)
                # Save the ISC data as a volume
                if not os.path.exists(f"{isc_map_path}/{task}"):
                    os.mkdir(f"{isc_map_path}/{task}")

                fn_nii = f"{sub[:6]}_{task}seg{idx:02d}_{kind}ISC.nii.gz"

                nib.save(isc_nifti, f"{isc_map_path}/{task}/{fn_nii}")


def _slice_img_timeseries(files, lng, affine=brain_nii.affine, roi=False):
    """Slice BOLD timeseries.

    Args:
        files (_type_): input files from each sub
        lng (_type_): length of time window
        affine (_type_, optional): shape of niimg. Defaults to brain_nii.affine.
        roi (bool, optional): whether or not in files are ROIs. Defaults to False.

    Returns:
        list of img_objs or list of 4D numpy.arrays: the former if voxel-based,
        the latter if ROIs is True. 
    """    
    masked_imgs = []
    sub_sliced = {}
    masker = NiftiLabelsMasker(mask_name)
    # Fetch images
    for i, processed in enumerate(files):
        if roi is False:
            timeserie = masker.fit_transform(processed)
            timeserie_len = timeserie.shape[3]
        else:
            timeserie = processed
            timeserie_len = int(len(timeserie))
        imgs_sub = []
        if lng == 100:
            range_step = range(0, int(timeserie_len - lng), int(lng / 2))
        else:
            range_step = range(0, int(timeserie_len - lng), lng)
        # slice them subject-wise
        for idx in range_step:
            slx = slice(0 + idx, lng + idx)
            if roi is False:
                sliced = nib.Nifti1Image(timeserie[:, :, :, slx], affine)
            else:
                sliced = timeserie[slx, :]
            # append sliced timeserie from 1 sub
            imgs_sub.append(sliced)
        # associate to key in dict for 1 sub
        sub_sliced[i] = imgs_sub
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
def map_isc(
    postproc_path,
    isc_map_path,
    kind="temporal",
    pairwise=False,
    roi=False,
    drop=None,
    slices=False,
    lng=100,
):
    """Main function to compute ISC maps

    Args:
        postproc_path (_type_): Path to processed data,
        input dir.
        isc_map_path (bool): Path to ISC maps data, output dir.
        kind (str, optional): kind of ISC to compute. Defaults to "temporal".
        pairwise (bool, optional): Approach to compute ISC. Defaults to False.
        roi (bool, optional): Whether or not the data is arranged in ROIs.
        Defaults to False.
        drop (str, optional): bids format sub ID to remove sub from analysis.
        Defaults to None.
        slices (bool, optional): whether or not to slice timeseries.
        Defaults to False.
        lng (int, optional): the length of time window to slice. Defaults to 100.
    """    
    # specify data path (leads to subdi
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {kind} ISC workflow")
    tasks = glob.glob(f"{postproc_path}/*/")

    # walks subdirs with taks name (task-s01-e01a)
    for idx_task, task in enumerate(sorted(tasks)):
        task = task[-13:-1]
        logger.info("Importing data")
        if roi is True:
            files = sorted(glob.glob(f"{postproc_path}/{task}/*256.npy"))
        else:
            files = sorted(glob.glob(f"{postproc_path}/{task}/*.nii.gz*"))

        if drop is None:
            logger.info("Considering all subjects for ISCs")
        else:
            fn_to_remove = fnmatch.filter(files, f"*{drop}*")
            logger.info(
                f"Not considering all subjects for ISCs \n Removing : {fn_to_remove}"
            )
            files.remove(fn_to_remove[0])
        for fn in files:
            _, fn = os.path.split(fn)
            logger.info(fn[:6])

        # do not compute if less than totality of considered subs
        if len(files) < 2:
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
            masked_imgs = _slice_img_timeseries(files, lng)

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
                if roi is False:
                    for niimg_obj in masked_imgs:
                        bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
                            niimg_obj, len(files)
                        )
                        isc_seg = isc(bold_imgs, pairwise=pairwise)
                        isc_imgs.append(isc_seg)
                else:
                    for bold_ts in masked_imgs:
                        isc_seg = isc(bold_ts, pairwise=pairwise)
                        isc_imgs.append(isc_seg)
        elif kind == "spatial":
            isc_imgs = isfc(bold_imgs, pairwise=pairwise)
        else:
            logger.info(f"Cannot compute {kind} ISC on {task}")
            continue

        logger.info("Saving images")
        if pairwise is False:
            _save_sub_feature_img(isc_imgs, isc_map_path, task, kind, files, roi)
            # free up memory
            del bold_imgs, isc_imgs

        # if it's not pairwise
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
