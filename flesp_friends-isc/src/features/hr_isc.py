"""HR-ISC workflow."""
import logging
import fnmatch
import pickle
import click
from dotenv import find_dotenv, load_dotenv
import glob
from brainiak.isc import isc
import neurokit2 as nk
import pandas as pd

subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]

fetcher = pd.read_csv(
    "/scratch/flesp/physio_data/friends1/pilot_hr-isc.csv", index_col=0
)
tasks_list = pd.Series(pd.unique(fetcher["task"].unique())).dropna().sort_values()
ok_task = []
for task in tasks_list:
    tmp = fetcher[fetcher["task"] == task]
    if len(tmp) >= len(subjects):
        ok_task.append(task)


def _resampling_rr_to_tr(postproc_path, tasks=ok_task):
    """
    """
    resampled_intervals = {}
    globbed_files = glob.glob(f"{postproc_path}/*/*/*.json")
    # find all files related to each task
    for task in tasks:
        rr_data = pd.DataFrame(columns=subjects)
        nb_vol = fetcher["nb_vol"][fetcher["task"] == task].values[0]
        task_fnames = fnmatch.filter(globbed_files, f"*{task}*")

        # go through each subject's physio file
        for sub in subjects:
            fname = fnmatch.filter(task_fnames, f"*{sub}*")
            if fname == []:
                continue
            with open(fname[0], "rb") as opener:
                json_file_with_physio_info = pickle.load(opener)
            rr_intervals = json_file_with_physio_info["PPG_clean_rr_systole"]
            data = pd.Series(
                nk.signal_resample(rr_intervals, desired_length=int(nb_vol + 12))
            )
            rr_data[sub] = data[6:-6]

        # organize dictionary by tasks
        resampled_intervals[task] = rr_data

    return resampled_intervals


@click.command()
@click.argument("postproc_path", type=click.Path(exists=True))
@click.argument("isc_hr_path", type=click.Path(exists=True))
@click.option("--length", type=int)
def hr_isc(postproc_path, isc_hr_path, tasks=ok_task, length=30):
    """
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting HR-ISC workflow")

    # fetching and making hr data compatible with bold sampling freq
    hr_intervals_tr_aligned = _resampling_rr_to_tr(postproc_path, tasks=tasks)
    logger.info("fetched HR data")

    # Build a dataframe where task coefficients are organized segment by sub
    coeffs = pd.DataFrame(index=subjects)

    # iterate through tasks
    for task in hr_intervals_tr_aligned.keys():

        # Segment the run
        for i, window in range(0, len(hr_intervals_tr_aligned[task]), length):
            # 100 TR long window, overlap
            segment = hr_intervals_tr_aligned[task].loc[window:window + length]

            if len(segment) < length / 2:
                continue
            # computing HR-ISC
            hr_isc_r_values = isc(segment.values, pairwise=False)

            coeffs[f"{task}seg{i:02d}"] = hr_isc_r_values.flatten()

    coeffs.to_csv(f"{isc_hr_path}/hr_isc_segments{length}tr.csv")


if __name__ == "__main__":
    # NOTE: from command line `make_dataset input_data output_filepath`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    hr_isc()
