import re
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import index_img, load_img, resample_to_img
from neuroimager.utils.rbload import rbload_imgs


def symmetrize_image_data(image_data, round: int or False = False):
    mid_point = image_data.shape[0] // 2
    left_hemisphere = image_data[:mid_point, :, :]
    right_hemisphere = image_data[mid_point:, :, :]
    right_hemisphere_flipped = np.flip(right_hemisphere, axis=0)
    symmetrized_data = (left_hemisphere + right_hemisphere_flipped) / 2
    full_symmetrized_data = np.concatenate(
        (symmetrized_data, np.flip(symmetrized_data, axis=0)), axis=0
    )

    return full_symmetrized_data


def symmetrize_image_nifti(input_file, output_paths: str or list = None):
    """
    Symmetrize a NIfTI image across the mid-sagittal plane.
    Returns a NIfTI image object or a list of NIfTI image objects.
    The images are saved if output_paths is specified.

    """
    nifti_images = rbload_imgs(input_file)
    save = False
    if isinstance(output_paths, str):
        output_paths = [output_paths]
        save = True
    elif isinstance(output_paths, list):
        save = True
    if len(output_paths) != len(nifti_images):
        raise ValueError(
            "The number of output paths must match the number of input images."
        )
    symmetrized_nifti_images = []
    for i, nifti_image in enumerate(nifti_images):
        image_data = nifti_image.get_fdata()
        symmetrized_data = symmetrize_image_data(image_data)
        symmetrized_nifti_image = nib.Nifti1Image(
            symmetrized_data, nifti_image.affine, nifti_image.header
        )
        if save:
            nib.save(symmetrized_nifti_image, output_paths[i])
        if len(nifti_images) == 1:
            return symmetrized_nifti_image
        symmetrized_nifti_images.append(symmetrized_nifti_image)

    return symmetrized_nifti_images


def resample_masks(source_img_name, target_img_name, interpolation="continuous"):
    target_img = load_img(target_img_name)
    source_img = load_img(source_img_name)
    res_img = resample_to_img(
        source_img,
        target_img,
        interpolation=interpolation,
        copy=True,
        order="F",
        clip=False,
        fill_value=0,
        force_resample=False,
    )

    if interpolation != "nearest":
        data = res_img.get_fdata()
        data_rounded = data.round().astype(int)
        res_img_rounded = nib.Nifti1Image(data_rounded, res_img.affine, res_img.header)

        return res_img_rounded
    else:
        return res_img


# TODO: Add support for nilean atlas objects
def combine_atlases(atlases: list, output_path: str, return_combined_atlas=True):
    """
    This function combines multiple atlas files (in NIfTI format) into a single atlas file.
    Prioritize atlases in the atlas_paths list by placing them at the front.

    Parameters:
    atlas_paths (list): A list of file paths or nib.nifti1.Nifti1Image to the atlas files to be combined.
            THE ORDER OF THIS LIST MATTERS. As the function will assign the first
            non-zero value to a voxel if there is an overlap.
    output_path (str): The file path where the combined atlas file will be saved.
    return_combined_atlas (bool): If True, the combined atlas will be returned.

    Returns:
    combined_atlas. The combined atlas is returned if return_combined_atlas is True.

    Notes:
    - The input atlas files should have the same affine matrix and dimensions.
    - The function assumes that the input atlases have non-overlapping labels.
    - The combined atlas file will be saved in NIfTI format (.nii.gz).

    Example:
    atlas_path = './assets/masks/'
    files = os.listdir(atlas_path)
    atlas_paths = [os.path.join(atlas_path, file) for file in files]
    output_path = "./assets/output/combined_atlas.nii.gz"
    combine_atlases(atlas_paths, output_path)
    """
    print(
        """Prioritize atlases in the atlas_paths list by placing them at the front.
        The function will assign the first non-zero value to a voxel if there is an overlap."""
    )
    combined_data = None
    max_label = 0
    affine = None
    header = None

    for atlas_file in atlases:
        if isinstance(atlas_file, str):
            atlas = nib.load(atlas_file)
        elif isinstance(atlas_file, nib.nifti1.Nifti1Image):
            atlas = atlas_file
        else:
            raise TypeError(
                "atlas_paths must be a list of file paths or nib.nifti1.Nifti1Image"
            )
        if affine is None:
            affine = atlas.affine
            header = atlas.header
        else:
            if not (atlas.affine == affine).all():
                print(f"Error: Affine matrix mismatch in {atlases}")

        atlas_data = atlas.get_fdata().astype(np.int16)
        if combined_data is None:
            combined_data = atlas_data
        else:
            if atlas_data.shape != combined_data.shape:
                print(f"Error: Dimension mismatch in {atlases}")

            atlas_data[atlas_data > 0] += max_label
            combined_data = np.where(combined_data == 0, atlas_data, combined_data)
        max_label = int(combined_data.max())

    combined_atlas = nib.Nifti1Image(combined_data, affine, header)
    nib.save(combined_atlas, output_path)
    return combined_atlas


def combine_probabilistic_atlases(atlases: list, output_path: str, thresh=0.25):
    """
    This function combines multiple probabilistic atlases into a single atlas by selecting the region with the highest probability for each voxel. The resulting atlas is saved to the specified output path.
    The atlas files must be 4D or 3D probabilistic atlas. The order of the atlases determines the label in the output file.

    Parameters:
    atlas_paths (list): A list of file paths or nib.nifti1.Nifti1Image to the probabilistic atlases to be combined.
            Works for a single file.
    output_path (str): The file path where the combined atlas will be saved.
    thresh (float, optional): A threshold value between 0 and 1 used to create a mask for voxels with low probabilities.Default is 0.25.
        Please Note this threshold is relative to the combined probability not individual atlas probability.

    Returns:
    combined_atlas (nibabel.nifti1.Nifti1Image): The combined atlas as a Nifti1Image object.

    Raises:
    Prints an error message if there is an affine matrix mismatch or dimension mismatch in the input atlases.

    Example:
    combined_atlas = combine_probabilistic_atlases(["atlas1.nii.gz", "atlas2.nii.gz"], "combined_atlas.nii.gz", thresh=0.5)
    """
    combined_data = None
    affine = None
    header = None

    # Check if atlas_paths is a list
    if not isinstance(atlases, list):
        raise TypeError("atlas_paths must be a list even if you have only one file")

    # Check if output_path is a string
    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string")

    for atlas_file in atlases:
        if isinstance(atlas_file, str):
            atlas = nib.load(atlas_file)
        elif isinstance(atlas_file, nib.nifti1.Nifti1Image):
            atlas = atlas_file
        else:
            raise TypeError(
                "atlas_paths must be a list of file paths or nib.nifti1.Nifti1Image"
            )

        if affine is None:
            affine = atlas.affine
            header = atlas.header
        else:
            if not (atlas.affine == affine).all():
                print(f"Error: Affine matrix mismatch in {atlas_file}")

        atlas_data = atlas.get_fdata()

        if atlas_data.ndim == 3:
            # make a 3d prob atlas to 4d prob atlas
            atlas_data = np.expand_dims(atlas_data, axis=-1)
        elif atlas_data.ndim == 4:
            pass
        else:
            print(f"Error: Dimension mismatch in {atlas_file}")

        # make sure all the atlas is on the same scale
        # e.g. some range from 1-100, some range from 0-1
        atlas_data = atlas_data / atlas_data.max()

        if combined_data is None:
            combined_data = atlas_data
        else:
            # concat all the files to one 4d array
            combined_data = np.concatenate((combined_data, atlas_data), axis=3)

    # dim reduce the mask to 3d by summing up the 4th dimension
    dim_red_data = np.sum(combined_data, axis=3)
    mask = dim_red_data < thresh
    # Select the region with the highest probability for each voxel
    max_prob_region_atlas = np.argmax(combined_data, axis=3)
    max_prob_region_atlas += 1  # Add 1 to avoid 0 indexing
    # Apply the mask to the atlas
    max_prob_region_atlas[mask] = 0

    combined_atlas = nib.Nifti1Image(max_prob_region_atlas, affine, header)
    nib.save(combined_atlas, output_path)

    return combined_atlas


def filter_rois(
    atlas: str or nib.nifti1.Nifti1Image,
    rois: list,
    output_path: str,
    remove: bool = True,
):
    """
    This function extracts or exclude specified regions of interest (ROIs) from a given atlas image and saves it.
    This function uses 0 indexing (starts from first ROI) for both 3D and 4D atlases.

    Parameters:
    atlas (str or nib.nifti1.Nifti1Image): The input atlas image, either as a file path string or a nibabel image object.
    output_path (str): The file path where the resulting image will be saved.
    rois (list of int): A list of ROI indices to include or exclude.
    remove (bool, optional): If True, the specified ROIS will be excluded otherwise will be included. Default is True.

    Returns:
    new_atlas_img (nib.nifti1.Nifti1Image): The resulting image with the specified ROIs included or removed.

    Raises:
    ValueError: If the input 'atlas' is not a string or a nibabel image object.
    """
    if isinstance(atlas, str):
        atlas_img = nib.load(atlas)
    elif isinstance(atlas, nib.nifti1.Nifti1Image):
        atlas_img = atlas
    else:
        raise ValueError("atlas must be a string or a nibabel image")

    # Turn the specified ROIS into include list
    if remove:
        # Create a list of all ROI indices
        num_rois = atlas_img.shape[-1]
        all_roi_indices = list(range(num_rois))
        # Remove the specified ROIs from the list
        for roi in rois:
            all_roi_indices.remove(roi)
    elif not remove:
        all_roi_indices = rois
    else:
        raise ValueError("remove must be a boolean")
    all_roi_indices.sort()

    dims = len(atlas_img.shape)
    if dims == 4:
        # Create a new 4D image with only the specified ROIs included
        new_atlas_img = index_img(atlas_img, all_roi_indices)
        nib.save(new_atlas_img, output_path)
    elif dims == 3:
        # Create a boolean mask with the same shape as atlas_data
        atlas_data = atlas_img.get_fdata()
        mask = np.isin(atlas_data, all_roi_indices)
        new_atlas_data = np.where(mask, atlas_data, 0)
        new_atlas_img = nib.Nifti1Image(new_atlas_data, atlas_img.affine)
        nib.save(new_atlas_img, output_path)

    return new_atlas_img


def split_schafer_names(schafer_labels: str or pd.DataFrame):
    """
    This function reads a CSV file containing Schafer parcellation data and extracts the hemisphere, network, and subregion
    information from the ROI names. The extracted information is added as new columns to the DataFrame and the updated
    DataFrame is returned.

    Args:
        schafer_labels (str or pd.DataFrame): The path to the CSV file containing Schafer parcellation data.

    Returns:
        pd.DataFrame: A DataFrame containing the original Schafer parcellation data with additional columns for
                      'Hemisphere', 'Network', and 'Subregion' extracted from the ROI names.
    """
    if isinstance(schafer_labels, str):
        schafer = pd.read_csv(schafer_labels)
    elif isinstance(schafer_labels, pd.DataFrame):
        schafer = schafer_labels
    regex_pattern = r"(RH|LH)_([A-Za-z]+)(?:_([A-Za-z]+))?(?:_[A-Za-z]+)*"

    def parse_roi_name(roi_name):
        match = re.search(regex_pattern, roi_name)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None

    schafer[["Hemisphere", "Network", "Subregion"]] = (
        schafer["ROI Name"].apply(parse_roi_name).apply(pd.Series)
    )

    return schafer
