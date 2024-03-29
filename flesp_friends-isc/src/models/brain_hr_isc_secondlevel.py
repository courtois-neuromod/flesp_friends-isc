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
from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
import itertools

fsaverage = fetch_surf_fsaverage()
mask_name = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
brain_nii = nib.load(mask_name)
brain_mask = io.load_boolean_mask(mask_name)
difumo = fetch_atlas_difumo(dimension=256)


def create_model_input(isc_path, seg_len, pairwise=False, threshold=None):
    """Build data dictionaries."""
    logger = logging.getLogger(__name__)
    subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
    if pairwise is True:
        fname = "pairwise_isc_hr_coeffs-seg"
        pairs = []
        for pair in itertools.combinations(subjects, 2):
            pairs.append(pair[0] + "-" + pair[1])
        subjects = pairs
    else:
        fname = "isc_hr_coeffs-seg"
    hr_coeffs = pd.read_csv(
        f"/scratch/flesp/physio_data/{fname}{seg_len}.csv", index_col=0
    )
    logger.info(hr_coeffs.head())
    brain_isc_dict = {}
    hr_isc_dict = {}
    for sub in subjects:
        logger.info(f"Creating model input for {sub}")
        i = subjects.index(sub)
        sub_hr_coeffs = hr_coeffs.iloc[i].to_frame().T
        if threshold is not None:
            logger.info(f"applying threshold: {threshold}")
            sub_hr_coeffs = sub_hr_coeffs.where(
                sub_hr_coeffs.values > threshold
            ).dropna(axis=1)
        hr_brain_segment_list = []
        segments_to_remove = []
        second_level_input = []
        for directory in sorted(glob.glob(f"{isc_path}{seg_len}/*")):
            task_name = os.path.split(directory)[1]
            scan_segments_list = sorted(
                glob.glob((f"{isc_path}{seg_len}/{task_name}/{sub}*"))
            )
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
                            f"{isc_path}{seg_len}/{task_name}/"
                            f"{sub}*{task_name}seg"
                            f"{idx:02d}*"
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
@click.argument("isc_path", type=click.Path(exists=False))
@click.argument("out_dir", type=click.Path(exists=True))
@click.option("--seg_len", type=str)
@click.option("--pairwise", type=bool)
@click.option("--threshold", type=float)
@click.option("--roi", type=bool)
def compute_model_contrast(
    isc_path, out_dir, seg_len="30", pairwise=False, threshold=None, roi=False,
):
    """Compute and save HR-ISC regressed Brain-ISC maps"""
    logger = logging.getLogger(__name__)
    subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
    # defining pairs if not subjects
    if pairwise is True:
        fname = "pw_isc_hr_coeffs-seg"
        map_name = "pw_roi_segments"
        if threshold is not None:
            map_name += f"_threshold{str(threshold)}"
        pairs = []
        for pair in itertools.combinations(subjects, 2):
            pairs.append(pair[0] + "-" + pair[1])
        subjects = pairs
    elif threshold is not None and pairwise is False:
        map_name = f"segments_threshold{str(threshold)}"
    else:
        map_name = "segments"

    # list models coeffs and niimg filenames
    brain_isc_dict, hr_isc_dict = create_model_input(
        f"{isc_path}", seg_len, pairwise, threshold=threshold
    )
    logger.info("Created data dictionaries")
    # initializing results
    max_eff_size = pd.DataFrame(index=subjects)
    eff_size = []
    coords_size = []
    coords_var = []
    variance = []
    maskers = NiftiMapsMasker(difumo.maps).fit()
    for sub in subjects:
        design_matrix = make_second_level_design_matrix(
            hr_isc_dict[sub]["subject_label"], hr_isc_dict[sub]
        )
        if os.path.exists(f"{out_dir}/{map_name}_{seg_len}TRs") is False:
            os.mkdir(f"{out_dir}/{map_name}_{seg_len}TRs")
        plotting.plot_design_matrix(
            design_matrix,
            output_file=f"{out_dir}/{map_name}_{seg_len}TRs/{sub}_design-matrix.png",
        )
        logger.info('created design matrix')
        if roi is True:
            masked_imgs = []
            for filename in brain_isc_dict[sub]:
                vol = np.load(filename)
                masked_imgs.append(maskers.inverse_transform(vol))
                logger.info(f'masking {filename}')
            brain_isc_dict[sub] = masked_imgs

        model = SecondLevelModel(smoothing_fwhm=6, n_jobs=-1).fit(
            brain_isc_dict[sub], design_matrix=design_matrix
        )
        #n_voxels = np.sum(get_data(model.masker_.mask_img_))
        logger.info("fitted model")
        stat_map = model.compute_contrast("r_coeffs", output_type="all")       
        logger.info(f"Computed model contrast for {sub}")

        # model results
        x, y, z = plotting.find_xyz_cut_coords(stat_map["effect_size"])
        a, b, c = plotting.find_xyz_cut_coords(stat_map["effect_variance"])
        coords_size.append([x, y, z])
        coords_var.append([a, b, c])
        max_eff = stat_map["effect_size"].get_fdata().max()
        eff_size.append(max_eff)
        variance_range = [
            stat_map["effect_variance"].get_fdata().min(),
            stat_map["effect_variance"].get_fdata().max(),
        ]
        variance.append(variance_range)

        # Make the ISC output a volume
        thresholded_map, threshold = threshold_stats_img(
            stat_map["z_score"],
            alpha=0.05,
            height_control="fpr",
            cluster_threshold=30,
            two_sided=True,
        )
        fdr_map, fdr_threshold = threshold_stats_img(
            stat_map["z_score"],
            alpha=0.05,
            height_control="fdr",
            cluster_threshold=10,
            two_sided=True,
        )

        view = plotting.view_img_on_surf(
            thresholded_map, threshold=threshold, surf_mesh="fsaverage"
        )
        view_fdr = plotting.view_img_on_surf(
            fdr_map, threshold=fdr_threshold, surf_mesh="fsaverage"
        )
        
        nib.save(
            stat_map['p_value'],
            f"{out_dir}/{map_name}_{seg_len}TRs/{sub}_HR-Brain-ISC_p-map.nii.gz",
        )

        nib.save(
            thresholded_map,
            f"{out_dir}/{map_name}_{seg_len}TRs/{sub}_HR-Brain-ISC_zmap.nii.gz",
        )

        view.save_as_html(
            f"{out_dir}/{map_name}_{seg_len}TRs/{sub}_HR-Brain-ISC-fpr_surface_plot.html"
        )
        view_fdr.save_as_html(
            f"{out_dir}/{map_name}_{seg_len}TRs/{sub}_HR-Brain-ISC-fdr_surface_plot.html"
        )
        logger.info(f"Saved stat map for {sub}")

    # log effect sizes of models
    max_eff_size["Effect_sizes"] = eff_size
    max_eff_size["Effect_size_coords"] = coords_size
    max_eff_size["Variance_range"] = variance
    max_eff_size["Effect_variance_coords"] = coords_var
    max_eff_size.to_csv(
        f"{out_dir}/{map_name}_{seg_len}TRs/effect_sizes-coords_and_variance_range.csv"
    )
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