import re
import pandas as pd
import numpy as np
import nibabel as nib


def combine_atlases(atlas_paths:list, output_path:str, return_combined_atlas=True):
    """
    This function combines multiple atlas files (in NIfTI format) into a single atlas file.
    Prioritize atlases in the atlas_paths list by placing them at the front.

    Parameters:
    atlas_paths (list): A list of file paths to the atlas files to be combined.
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
    print("""Prioritize atlases in the atlas_paths list by placing them at the front.
        The function will assign the first non-zero value to a voxel if there is an overlap.""")
    combined_data = None
    max_label = 0
    affine = None
    header = None

    for atlas_path in atlas_paths:
        atlas = nib.load(atlas_path)
        print(atlas_path)
        if affine is None:
            affine = atlas.affine
            header = atlas.header
        else:
            if not (atlas.affine == affine).all():
                print(f"Error: Affine matrix mismatch in {atlas_path}")

        atlas_data = atlas.get_fdata().astype(np.int16)
        if combined_data is None:
            combined_data = atlas_data
        else:
            if atlas_data.shape != combined_data.shape:
                print(f"Error: Dimension mismatch in {atlas_path}")

            atlas_data[atlas_data > 0] += max_label
            combined_data = np.where(combined_data == 0, atlas_data, combined_data)
        print("atlas",pd.Series(atlas_data.flatten()).value_counts())
        max_label = int(combined_data.max())
        print("max",max_label)

    combined_atlas = nib.Nifti1Image(combined_data, affine, header)
    nib.save(combined_atlas, output_path)
    return combined_atlas


def split_schafer_names(file_path):
    """
    This function reads a CSV file containing Schafer parcellation data and extracts the hemisphere, network, and subregion
    information from the ROI names. The extracted information is added as new columns to the DataFrame and the updated
    DataFrame is returned.

    Args:
        file_path (str): The path to the CSV file containing Schafer parcellation data.

    Returns:
        pd.DataFrame: A DataFrame containing the original Schafer parcellation data with additional columns for
                      'Hemisphere', 'Network', and 'Subregion' extracted from the ROI names.
    """
    schafer = pd.read_csv(file_path)
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
