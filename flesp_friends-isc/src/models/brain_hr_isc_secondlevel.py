"""Brain-HR-ISC workflow."""
import os
import logging
import glob
import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import nibabel as nib
from brainiak import io

# niimg
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix

subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]

hr_coeffs = pd.read_csv("/scratch/flesp/data/hr_isc_coeffs_segments100TR.csv", index_col=0)

dirs = glob.glob("/scratch/flesp/data/isc-segment/*")

mask_name = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
brain_nii = nib.load(mask_name)
brain_mask = io.load_boolean_mask(mask_name)
coords = np.where(brain_mask)


def create_model_input(isc_path,):
    """Build data dictionaries."""
    logger = logging.getLogger(__name__)

    brain_isc_dict = {}
    hr_isc_dict = {}
    for sub in subjects:
        logger.info(f"Creating model input for {sub}")
        sub_hr_coeffs = hr_coeffs.iloc[subjects.index(sub)].to_frame().T
        hr_brain_segment_list = []
        segments_to_remove = []
        second_level_input = []

        for dir in sorted(glob.glob(f"{isc_path}/*")):

            task_name = os.path.split(dir)[1]
            scan_segments_list = sorted(glob.glob(f"{isc_path}/{task_name}/{sub}*"))
            # Make sure subject has proper files
            if scan_segments_list == []:
                logger.info(f"no file for {sub} in {task_name}")
                continue
            # Make sure the scan has concurrent HR data
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
    """Compute and save HR-ISC regressed Brain-ISC maps"""
    logger = logging.getLogger(__name__)
    brain_isc_dict, hr_isc_dict = create_model_input(isc_path)
    logger.info("Created data dictionaries")

    for sub in subjects:
        design_matrix = make_second_level_design_matrix(
            hr_isc_dict[sub]["subject_label"], hr_isc_dict[sub]
        )
        model = SecondLevelModel(smoothing_fwhm=6).fit(
            brain_isc_dict[sub], design_matrix
        )
        z_score_map = model.compute_contrast("r_coeffs", output_type="z_score")
        logger.info(f"Computed model contrast for {sub}")

        # Make the ISC output a volume
        isc_vol = np.zeros(brain_nii.shape)
        isc_vol[coords] = z_score_map
        isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)
        fn = f"{sub}_HR-Brain-ISC.nii.gz"
        nib.save(isc_nifti, f"{isc_path}/{fn}")
        logger.info(f"Saved stat map for {sub}")
    logger.info(f"Done workflow \n _______________________")

if __name__ == "__main__":
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    compute_model_contrast()
