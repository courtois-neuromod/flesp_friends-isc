"""Brain-HR-ISC workflow."""
import logging
import fnmatch
import pickle
import click
import itertools
from dotenv import find_dotenv, load_dotenv
import glob
import pandas as pd

# niimg
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
import neurokit2 as nk

subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]

hr_coeffs = pd.read_csv("data/isc_hr_coeffs-seg.csv", index_col=0)

dirs = glob.glob("data/iscs/*")


def create_model_input(isc_path,):
    """
    """
    logger = logging.getLogger(__name__)

    brain_isc_dict = {}
    hr_isc_dict = {}
    for sub in subjects:
        logger.info(f"Creating model input for {sub}")
        sub_hr_coeffs = hr_coeffs.iloc[subjects.index(sub)]
        hr_brain_segment_list = []
        segments_to_remove = []
        second_level_input = []

        for dir in sorted(glob.glob(f"{isc_path}/*")):

            task_name = os.path.split(dir)[1]
            scan_segments_list = sorted(glob.glob(f"{isc_path}/{task_name}/{sub}*"))

            if scan_segments_list == []:
                logger.info(f"no file for {sub} in {task_name}")
                continue

            for idx in range(len(scan_segments_list)):
                if f"{task_name}seg{idx:02d}" in sub_hr_coeffs.columns:
                    hr_brain_segment_list.append(f"{task_name}seg{idx:02d}")
                    second_level_input.append(
                        glob.glob(
                            f"{isc_path}/{task_name}/"
                            f"{sub}*{task_name}seg"
                            f"{idx:02d}_"
                            f"temporalISC.nii.gz"
                        )[0]
                    )
            for segment in sub_hr_coeffs.columns:
                if segment not in hr_brain_segment_list:
                    segments_to_remove.append(segment)
        brain_isc_dict[sub] = sorted(second_level_input)
        hr_isc_dict[sub] = pd.DataFrame(
            {
                "subject_label": sorted(hr_brain_segment_list),
                "r_coeffs": sub_hr_coeffs.drop(columns=segments_to_remove),
            }
        )

        logger.info(
            f"nb of retained episode segments :"
            f"{len(hr_brain_segment_list)}\n"
            f"total number of episode segments : "
            f"{len(sub_hr_coeffs.columns)}"
        )
    return brain_isc_dict, hr_isc_dict


@click.command()
@click.argument("isc_path", type=click.Path(exists=True))
def compute_model_contrast(isc_path,):
    """
    """
    logger = logging.getLogger(__name__)

    brain_isc_dict, hr_isc_dict = create_model_input(isc_path)

    for sub in subjects:
        design_matrix = make_second_level_design_matrix(
            hr_isc_dict[sub]["subject_label"], hr_isc_dict[sub]
        )
        model = SecondLevelModel(smoothing_fwhm=6).fit(
            brain_isc_dict[sub], design_matrix
        )
        z_score_map = model.compute_contrast('r_coeffs', output_type='z_score')
