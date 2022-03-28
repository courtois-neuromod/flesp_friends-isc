import numpy as np

from brainiak.isc import isc
from brainiak import io, image
import nibabel as nib

project_dir = Path(__file__).resolve().parents[2]
subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']

# make sure it's installed
mask_name = 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
brain_mask = io.load_boolean_mask(mask_name)
coords = np.where(brain_mask)
brain_nii = nib.load(mask_name)


def compute_bold_isc(episodes_path, pairwise=pairwise):
    """
    Compute ISC for brain data.

    note
    """
    # specify data path (leads to subdirs)
    MultiSubject_episode_path = f'{project_dir}{episodes_path}'
    # walks subdirs with taks name (task-s01-e01a)
    for task in glob.glob(MultiSubject_episode_path):
        files = sorted(glob.glob(f'{task}*fwhm*_bold.*'))

        images = io.load_images(files)
        masked_imgs = image.mask_images(images, brain_mask)

        # compute ISC
        bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
            masked_imgs, 6)
        bold_imgs[np.isnan(bold_imgs)] = 0
        isc_imgs = isc(bold_imgs, pairwise=pairwise)

        # save ISC maps per subject
        for n, subj in enumerate(subjects):
            # Make the ISC output a volume
            isc_vol = np.zeros(brain_nii.shape)
            # Map the ISC data for the first participant into brain space
            isc_vol[coords] = isc_imgs[n, :]
            # make a nii image of the isc map
            isc_nifti = nib.Nifti1Image(
                isc_vol, brain_nii.affine, brain_nii.header
            )

            # Save the ISC data as a volume
            isc_map_path = f'ISC_{task}_{subj}.nii.gz'
            nib.save(isc_nifti, isc_map_path)

        # free up memory
        del bold_imgs, isc_imgs
