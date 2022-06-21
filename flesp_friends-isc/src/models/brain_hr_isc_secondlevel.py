"""Brain-HR-ISC workflow."""
import os
import logging
import glob
import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

# niimg
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
import nibabel as nib
from brainiak import io
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage

subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
seg_len = "30"
hr_coeffs = pd.read_csv(
    f"/scratch/flesp/physio_data/isc_hr_coeffs-seg{seg_len}.csv", index_col=0
)

dirs = glob.glob(f"/scratch/flesp/data/isc-segments{seg_len}/*")

fsaverage = fetch_surf_fsaverage()
mask_name = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
brain_nii = nib.load(mask_name)
brain_mask = io.load_boolean_mask(mask_name)
coords = np.where(brain_mask)


def create_model_input(
    isc_path,
):
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
                "r_coeffs": sub_hr_coeffs.drop(columns=segments_to_remove)
                .iloc[0]
                .squeeze(),
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
@click.argument("out_dir", type=click.Path(exists=True))
def compute_model_contrast(
    isc_path,
    out_dir,
):
    """Compute and save HR-ISC regressed Brain-ISC maps"""
    logger = logging.getLogger(__name__)
    brain_isc_dict, hr_isc_dict = create_model_input(isc_path)
    logger.info("Created data dictionaries")
    max_eff_size = pd.DataFrame(index=subjects)
    eff_size = []

    for sub in subjects:
        design_matrix = make_second_level_design_matrix(
            hr_isc_dict[sub]["subject_label"], hr_isc_dict[sub]
        )
        plotting.plot_design_matrix(
            design_matrix,
            output_file=f"{out_dir}/segments{seg_len}TRs/{sub}_design-matrix.png",
        )
        logger.info("created design matrix")
        model = SecondLevelModel(smoothing_fwhm=6).fit(
            brain_isc_dict[sub], design_matrix=design_matrix
        )
        logger.info("fitted model")
        stat_map = model.compute_contrast("r_coeffs", output_type="all")
        logger.info(f"Computed model contrast for {sub}")
        max = stat_map["effect_size"].get_fdata().max()
        eff_size.append(max)

        # Make the ISC output a volume
        thresholded_map, threshold = threshold_stats_img(
            z_score_map,
            alpha=0.05,
            height_control="fpr",
            cluster_threshold=10,
            two_sided=True,
        )
        view = plotting.view_img_on_surf(
            thresholded_map, threshold=threshold, surf_mesh="fsaverage"
        )

        nib.save(
            stat_map["z_score"],
            f"{out_dir}/segments{seg_len}TRs/{sub}_HR-Brain-ISC_zmap.nii.gz",
        )

        view.save_as_html(
            f"{out_dir}/segments{seg_len}TRs/{sub}_HR-Brain-ISC_surface_plot.html"
        )
        logger.info(f"Saved stat map for {sub}")

    # log effect sizes of models
    max_eff_size["Effect_sizes"] = eff_size
    max_eff_size.to_csv(f"{out_dir}/segments{seg_len}TRs/max_effect_sizes.csv")
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
